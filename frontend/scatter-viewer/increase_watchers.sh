#!/bin/bash
# Temporarily increase file watcher limit for current session
echo "Current file watcher limit: $(cat /proc/sys/fs/inotify/max_user_watches)"
echo "Increasing to 524288..."
sudo sysctl fs.inotify.max_user_watches=524288
echo "New limit: $(cat /proc/sys/fs/inotify/max_user_watches)"
echo ""
echo "Note: This is temporary and will reset on reboot."
echo "To make it permanent, add 'fs.inotify.max_user_watches=524288' to /etc/sysctl.conf"