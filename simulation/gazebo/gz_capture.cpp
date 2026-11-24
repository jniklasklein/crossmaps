// Local-only ABI boundary: Gazebo's Homebrew C++ dependencies never enter the
// existing ROS/Python environment. stdout is JSONL, stderr is diagnostics.
#include <gz/transport/Node.hh>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/camera_info.pb.h>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/laserscan.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <google/protobuf/util/json_util.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

namespace {
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> running{true};
std::mutex pendingMutex;
std::map<std::string, std::string> pending;
void stop(int) { running = 0; }
template<class Message> void receive(const std::string &kind, const Message &msg) {
  std::string json;
  google::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  const auto result = google::protobuf::util::MessageToJsonString(msg, &json, options);
  if (!result.ok()) { std::cerr << "Cannot serialize " << kind << '\n'; return; }
  std::lock_guard<std::mutex> lock(pendingMutex);
  // At most one outstanding packet per stream; slow consumers cannot build a
  // camera backlog. ROS adapter independently pairs identical capture stamps.
  pending[kind] = "{\"kind\":\"" + kind + "\",\"message\":" + json + "}\n";
}
}

int main(int argc, char **argv) {
  if (argc < 2 || (std::string(argv[1]) != "rover_a" && std::string(argv[1]) != "rover_b" && std::string(argv[1]) != "evaluation")) {
    std::cerr << "Usage: gz_capture rover_a|rover_b|evaluation [clock_topic]\n"; return 2;
  }
  // Never attach to the default/global simulator transport partition.
  const char *partition = std::getenv("GZ_PARTITION");
  if (!partition || std::string(partition).rfind("crossmaps_sim_", 0) != 0) {
    std::cerr << "GZ_PARTITION must start with crossmaps_sim_\n"; return 2;
  }
  std::signal(SIGINT, stop); std::signal(SIGTERM, stop); std::signal(SIGPIPE, SIG_IGN);
  gz::transport::Node node;
  if (std::string(argv[1]) == "evaluation") {
    // Read-only scoring channel. No sensor, odometry, TF or command publisher
    // exists in this mode, and ground truth never enters the ROS bridges.
    bool ok = true;
    for (const std::string robot : {"rover_a", "rover_b"}) {
      ok &= node.Subscribe<gz::msgs::Pose_V>("/model/"+robot+"/pose",
        [robot](const gz::msgs::Pose_V &m){
          if (m.pose_size() > 8) return;
          receive(robot,m);
        });
    }
    ok &= node.Subscribe<gz::msgs::Clock>("/clock", [kind=std::string("clock")](const gz::msgs::Clock &m){receive(kind,m);});
    if (!ok) { std::cerr << "Ground-truth subscriptions failed\n"; return 3; }
    while (running) {
      std::map<std::string,std::string> batch;
      { std::lock_guard<std::mutex> lock(pendingMutex); batch.swap(pending); }
      for (const auto &packet : batch) std::cout << packet.second;
      std::cout.flush();
      if (!std::cout) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return 0;
  }
  const std::string root = "/" + std::string(argv[1]);
  const std::string clockTopic = argc > 2 ? argv[2] : "/clock";
  bool ok = true;
  ok &= node.Subscribe<gz::msgs::Clock>(clockTopic, [kind=std::string("clock")](const gz::msgs::Clock &m){receive(kind,m);});
  ok &= node.Subscribe<gz::msgs::Image>(root+"/rgbd/image", [kind=std::string("rgb")](const gz::msgs::Image &m){receive(kind,m);});
  ok &= node.Subscribe<gz::msgs::Image>(root+"/rgbd/depth_image", [kind=std::string("depth")](const gz::msgs::Image &m){receive(kind,m);});
  ok &= node.Subscribe<gz::msgs::CameraInfo>(root+"/rgbd/camera_info", [kind=std::string("info")](const gz::msgs::CameraInfo &m){receive(kind,m);});
  ok &= node.Subscribe<gz::msgs::Odometry>(root+"/odom", [kind=std::string("odom")](const gz::msgs::Odometry &m){receive(kind,m);});
  ok &= node.Subscribe<gz::msgs::LaserScan>(root+"/scan", [kind=std::string("scan")](const gz::msgs::LaserScan &m){
    // Fixed-size planar sensor only; reject malformed/unbounded payloads before
    // JSON encoding. receive() retains only the latest pending scan.
    if (m.count() < 2 || m.count() > 4096 || m.vertical_count() > 1 ||
        m.ranges_size() != static_cast<int>(m.count()) ||
        (m.intensities_size() != 0 && m.intensities_size() != m.ranges_size())) {
      std::cerr << "Rejected malformed planar lidar packet\n"; return;
    }
    receive(kind,m);
  });
  auto publisher = node.Advertise<gz::msgs::Twist>(root+"/cmd_vel");
  if (!ok || !publisher) { std::cerr << "Gazebo subscription/advertisement failed\n"; return 3; }

  // Independent wall-clock thread: blocked camera output cannot defeat the
  // deadman. stdin commands are plain 'forward_m_s yaw_rad_s\n'. EOF stops.
  std::thread control([&]{
    using Clock = std::chrono::steady_clock;
    auto last = Clock::now() - std::chrono::seconds(1);
    double forward = 0., yaw = 0.;
    std::string buffer;
    while (running) {
      pollfd fd{STDIN_FILENO, POLLIN, 0};
      const int ready = poll(&fd, 1, 50);
      if ((ready < 0 && errno != EINTR) || (ready > 0 && (fd.revents & (POLLERR | POLLNVAL)))) {
        running = 0; break;
      }
      if (ready > 0 && (fd.revents & (POLLIN | POLLHUP))) {
        char input[256]; const auto count = read(STDIN_FILENO,input,sizeof(input));
        if (count <= 0) { running = 0; break; }
        buffer.append(input, static_cast<size_t>(count));
        if (buffer.size() > 1024) { buffer.clear(); forward=yaw=0.; }
        size_t end;
        while ((end=buffer.find('\n')) != std::string::npos) {
          std::istringstream command(buffer.substr(0,end)); buffer.erase(0,end+1);
          double v,w; std::string extra;
          if ((command>>v>>w) && !(command>>extra) && std::isfinite(v) && std::isfinite(w)) {
            forward=std::clamp(v,-.35,.35); yaw=std::clamp(w,-.7,.7); last=Clock::now();
          } else { forward=yaw=0.; }
        }
      }
      const bool fresh=Clock::now()-last < std::chrono::milliseconds(500);
      gz::msgs::Twist twist;
      twist.mutable_linear()->set_x(fresh ? forward : 0.);
      twist.mutable_angular()->set_z(fresh ? yaw : 0.);
      publisher.Publish(twist);
    }
    gz::msgs::Twist zero; publisher.Publish(zero);
  });
  while (running) {
    std::map<std::string,std::string> batch;
    { std::lock_guard<std::mutex> lock(pendingMutex); batch.swap(pending); }
    for (const auto &packet:batch) std::cout << packet.second;
    std::cout.flush();
    if (!std::cout) { running=0; break; }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  running=0; control.join(); return 0;
}
