"""Offline auth paths. Never read/write someone's Keychain, SSH config or robots."""
import io
import ctypes as c
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import collab_relay_auth as auth


def robots():
    return [dict(id=name,camera_relay=dict(mode='auto',ssh_host=name+'@example.invalid',container='test'))
            for name in ('a','b')]


class AuthTests(unittest.TestCase):
    def prepare(self, check, **kw):
        log = []
        result = auth.prepare(robots(), verify=check, report=log.append, **kw)
        self.assertNotIn('test-secret',' '.join(log))
        return result, log

    def test_plain_launch_prefers_working_keys_without_opening_keychain_items(self):
        store, prompt = Mock(), Mock()
        self.assertEqual(self.prepare(lambda *_:'ok',store=store,prompt=prompt)[0],{})
        store.get.assert_not_called(); prompt.assert_not_called()

    def test_shared_prompt_enrolls_only_needy_host_after_proof(self):
        store, prompt = Mock(), Mock(return_value='test-secret')
        result, _ = self.prepare(lambda r,p:'ok' if r['id']=='a' or p=='test-secret' else 'denied',
                                 setup=True,interactive=True,store=store,prompt=prompt)
        self.assertEqual(result,{'b':'test-secret'})
        store.put.assert_called_once_with('b@example.invalid','test-secret')
        self.assertNotIn('a@example.invalid',prompt.call_args.args[0])

    def test_enrollment_never_writes_after_any_verification_failure(self):
        store=Mock()
        with self.assertRaisesRegex(ValueError,'no password saved'):
            self.prepare(lambda r,p:'ok' if r['id']=='a' and p else 'denied',setup=True,
                         interactive=True,store=store,prompt=lambda _:'test-secret')
        store.put.assert_not_called()

    def test_keychain_is_scoped_per_host_and_session_secret_is_not_logged_or_written(self):
        store=Mock(); store.get.side_effect=lambda host:host.split('@')[0]+'test-secret'
        passwords, _ =self.prepare(lambda r,p:'ok' if p==r['id']+'test-secret' else 'denied',store=store)
        self.assertEqual(passwords,{'a':'atest-secret','b':'btest-secret'})
        store.put.assert_not_called()

    def test_normal_interactive_launch_prompts_without_flag_but_does_not_persist(self):
        store=Mock(); store.get.return_value=None
        result,_=self.prepare(lambda r,p:'ok' if p else 'denied',interactive=True,store=store,prompt=lambda _:'test-secret')
        self.assertEqual(result,{'a':'test-secret','b':'test-secret'})
        store.put.assert_not_called()

    def test_default_enrollment_does_not_send_one_robots_password_to_another(self):
        store=Mock()
        def private_prompt(question):
            return 'a-test-secret' if 'a@example.invalid' in question else 'b-test-secret'
        def check(r,p):
            if not p: return 'denied'
            self.assertEqual(p,r['id']+'-test-secret')
            return 'ok'
        self.prepare(check,interactive=True,setup=True,store=store,prompt=private_prompt)
        self.assertEqual(store.put.call_count,2)

    def test_network_loss_during_cached_password_check_does_not_block_start(self):
        store=Mock();store.get.return_value='test-secret'
        result,log=self.prepare(lambda r,p:'offline' if p else 'denied',store=store)
        self.assertEqual(len(result),2);self.assertTrue(all('unverified' in row for row in log))

    def test_unattended_denied_fails_early_instead_of_silent_downgrade(self):
        with self.assertRaisesRegex(ValueError,'--setup-relay-auth'):
            self.prepare(lambda *_:'denied')

    def test_unreachable_source_does_not_block_peer_and_keeps_its_saved_secret(self):
        store=Mock(); store.get.return_value='test-secret'
        result,log=self.prepare(lambda r,p:'ok' if r['id']=='b' else 'offline',store=store)
        self.assertEqual(result,{'a':'test-secret'})
        self.assertIn('unverified',log[0])

    def test_host_key_failure_never_sends_a_password(self):
        calls=[]
        def check(r,p): calls.append(p); return 'host_key'
        with self.assertRaisesRegex(ValueError,'host_key'):
            self.prepare(check,prompt=Mock(side_effect=AssertionError),session_password='test-secret')
        self.assertEqual(calls,[None,None])

    def test_sshpass_receives_only_private_fd_no_secret_in_command_or_environment(self):
        def inspect(args,**kw):
            self.assertEqual(args[:2],['sshpass','-d'])
            self.assertNotIn('test-secret',repr(args)+repr(kw['env']))
            fd=int(args[2]); self.assertEqual(kw['pass_fds'],(fd,))
            self.assertEqual(os.read(fd,32),b'test-secret\n')
            return True
        with patch.object(auth.subprocess,'Popen',side_effect=inspect):
            self.assertTrue(auth.launch_ssh(['ssh','example'],{'SSHPASS':'test-secret',
                auth.PASSWORD_ENV:'test-secret'},'test-secret'))

    def test_dashboard_demo_check_or_replay_do_not_require_credentials(self):
        import crossmaps_collaborative as cli
        with patch.object(auth,'prepare',side_effect=AssertionError('should not access credentials')),\
             patch.object(cli,'check_environment'),patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(cli.main(['--demo','--check']),0)

    def test_no_password_or_gui_prompt_when_all_relay_modes_are_off(self):
        self.assertEqual(auth.prepare([], verify=Mock(side_effect=AssertionError)),{})
        with self.assertRaisesRegex(ValueError,'No relay'):
            auth.prepare([],setup=True)

    @staticmethod
    def mock_keychain():
        kc=object.__new__(auth.MacKeychain); kc.sec=Mock(); kc.cf=Mock()
        def previous(out):
            c.cast(out,c.POINTER(c.c_ubyte))[0]=1;return 0
        def default(out):
            c.cast(out,c.POINTER(c.c_void_p))[0]=77;return 0
        kc.sec.SecKeychainGetUserInteractionAllowed.side_effect=previous
        kc.sec.SecKeychainSetUserInteractionAllowed.return_value=0
        kc.sec.SecKeychainCopyDefault.side_effect=default
        return kc

    def test_native_keychain_absent_and_locked_do_not_invoke_ui_or_leak_ref(self):
        kc=self.mock_keychain()
        kc.sec.SecKeychainFindGenericPassword.return_value=-25300
        self.assertIsNone(kc.get('a@host'))
        kc.sec.SecKeychainFindGenericPassword.return_value=-25308
        with self.assertRaisesRegex(auth.KeychainError,'unlock'):
            kc.get('a@host')
        self.assertEqual(kc.cf.CFRelease.call_count,2)
        values=[x.args[0] for x in kc.sec.SecKeychainSetUserInteractionAllowed.call_args_list]
        self.assertEqual([(v if isinstance(v,bool) else v.value) for v in values],[False,1,False,1])

    def test_native_keychain_rotate_and_read_copies_and_frees(self):
        kc=self.mock_keychain(); native=c.create_string_buffer(b'test-secret')
        def find(store,nservice,service,naccount,account,n,ptr,item):
            self.assertEqual((service,account),(auth.SERVICE,b'a@host'))
            if item is not None:
                c.cast(item,c.POINTER(c.c_void_p))[0]=99
            else:
                c.cast(n,c.POINTER(c.c_uint32))[0]=11
                c.cast(ptr,c.POINTER(c.c_void_p))[0]=c.addressof(native)
            return 0
        kc.sec.SecKeychainAddGenericPassword.return_value=-25299
        kc.sec.SecKeychainFindGenericPassword.side_effect=find
        kc.sec.SecKeychainItemModifyAttributesAndData.return_value=0
        kc.put('a@host','test-secret')
        self.assertEqual(kc.sec.SecKeychainItemModifyAttributesAndData.call_args.args[-1],b'test-secret')
        self.assertEqual(kc.get('a@host'),'test-secret')
        kc.sec.SecKeychainItemFreeContent.assert_called_once()
