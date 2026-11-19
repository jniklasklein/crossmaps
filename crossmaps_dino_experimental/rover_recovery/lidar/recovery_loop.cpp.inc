  // CrossMaps LD19 recovery v1. Keep this ROS publisher alive across USB resets.
  auto publisher = node->create_publisher<sensor_msgs::msg::LaserScan>(topic_name, 10);
  using Clock = std::chrono::steady_clock;
  using namespace std::chrono_literals;
  int retry_seconds = 1;
  while (rclcpp::ok()) {
    ldlidar::LDLidarDriver driver;
    driver.RegisterGetTimestampFunctional(std::bind(&GetSystemTimeStamp));
    driver.EnableFilterAlgorithnmProcess(true);
    if (driver.Start(type_name, port_name, serial_port_baudrate, ldlidar::COMM_SERIAL_MODE)) {
      bool first_scan = true;
      auto last_valid = Clock::now();
      auto connected = last_valid;
      ldlidar::Points2D points;
      // More startup grace for sensor spin-up, then a shorter loss threshold.
      while (rclcpp::ok()) {
        double hz = 0.0;
        if (driver.GetLaserScanData(points, 1500) == ldlidar::LidarStatus::NORMAL &&
            points.size() > 1 && driver.GetLidarScanFreq(hz) && hz > 0.0) {
          if (first_scan) RCLCPP_INFO(node->get_logger(), "LD19 receiving valid scans from %s", port_name.c_str());
          ToLaserscanMessagePublish(points, hz, setting, node, publisher, first_scan);
          first_scan = false;
          last_valid = Clock::now();
          if (last_valid - connected >= 10s) retry_seconds = 1;
        } else if (Clock::now() - last_valid >= (first_scan ? 8s : 5s)) {
          RCLCPP_WARN(node->get_logger(), "No valid LD19 scans on %s; closing this connection", port_name.c_str());
          break;
        }
        rclcpp::sleep_for(100ms);
      }
    }
    driver.Stop();  // Join the receiver before destroying its packet parser.
    if (!rclcpp::ok()) break;
    RCLCPP_WARN(node->get_logger(), "Retrying LD19 %s in %d s", port_name.c_str(), retry_seconds);
    // Poll shutdown even if the ROS context changes while entering the wait.
    const auto retry_at = Clock::now() + std::chrono::seconds(retry_seconds);
    while (rclcpp::ok() && Clock::now() < retry_at) rclcpp::sleep_for(100ms);
    retry_seconds = std::min(15, retry_seconds * 2);
  }
