#!/usr/bin/env bash
# run_distance_test.sh - 引导式距离误差数据采集.
#
# 前提: 完整系统已用 enable_distance_recorder:=true 启动, 即:
#   ros2 launch apriltag_pose apriltag_pose.launch.py \
#       enable_distance_recorder:=true use_rviz:=false
#
# 本脚本循环遍历真实距离 x 姿态分组, 提示你摆放标签, 然后调用
# /distance_recorder_node/start_batch 服务采集样本 (追加写入 CSV).
#
# 用法: bash run_distance_test.sh
set -u
source /opt/ros/humble/setup.bash 2>/dev/null

# 实验配置 (与 config/system.yaml 保持一致; 也可按需修改)
DISTANCES="0.30 0.50 0.70 1.00"
GROUPS="front tilted"
SAMPLES="${SAMPLES:-20}"

echo "============================================================"
echo " 距离误差数据采集 (每个 距离x姿态 采集 ${SAMPLES} 个样本)"
echo " 请确保 distance_recorder_node 已随系统启动."
echo "============================================================"

for d in $DISTANCES; do
  for g in $GROUPS; do
    echo
    echo "------------------------------------------------------------"
    echo " 请把 AprilTag 放在距相机光心 ${d} m 处, 姿态: ${g}"
    echo "   front  = 正对相机 (标签平面垂直于光轴)"
    echo "   tilted = 轻微倾斜 (约 20~30 度)"
    echo " 用卷尺测量相机光心到标签中心的距离, 确认接近 ${d} m 后回车."
    read -r -p " >>> 摆放就绪后按回车开始采集 ... " _

    echo " >>> 设置 true_distance_m=${d} sample_group=${g} ..."
    ros2 param set /distance_recorder_node true_distance_m "$d" 2>/dev/null
    ros2 param set /distance_recorder_node sample_group "$g" 2>/dev/null
    ros2 param set /distance_recorder_node sample_count "$SAMPLES" 2>/dev/null

    echo " >>> 触发 start_batch ..."
    ros2 service call /distance_recorder_node/start_batch std_srvs/srv/Trigger 2>/dev/null
    echo " >>> 等待采集完成 (节点日志会打印 BATCH COMPLETE) ..."
  done
done

echo
echo "============================================================"
echo " 全部采集完成. 运行分析脚本生成报告:"
echo "   python3 \$(ros2 pkg prefix apriltag_pose)/share/apriltag_pose/scripts/analyze_distance_results.py"
echo " 或:"
echo "   python3 ~/apriltag_pose_ws/src/apriltag_pose/scripts/analyze_distance_results.py"
echo "============================================================"
