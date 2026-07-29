# Phase B 시뮬레이션 이미지 (spec §2.4, B-2a)
# ros:jazzy (arm64 네이티브) + Gazebo Harmonic (ros_gz 공식 페어링)
FROM ros:jazzy

RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-jazzy-ros-gz \
    && rm -rf /var/lib/apt/lists/*

# 기본 진입: ROS2 환경 소싱
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
