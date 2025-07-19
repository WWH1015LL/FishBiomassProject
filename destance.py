import cv2
import numpy as np
import os

# 获取当前工作目录并拼接视频路径（修正路径拼接方式）
pwd = os.getcwd()
video_path = os.path.join(pwd, "vedio", "biomass_m_1.mp4")

# 检查视频文件是否存在
if not os.path.exists(video_path):
    print(f"错误：视频文件不存在于 {video_path}")
    exit()

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("错误：无法打开视频文件")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
half_width = frame_width // 2

# 创建输出视频
output_path = os.path.join(pwd+r"\vedio", "output_biomass_m_1.mp4")
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (half_width, frame_height),
)

# 初始化视差计算器
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=9,
    P1=8 * 3 * 9**2,
    P2=32 * 3 * 9**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# 重投影矩阵 Q
Q = np.float32([[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 800], [0, 0, 1 / 0.1, 0]])

# 创建显示窗口
cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stereo Depth", 1280, 480)  # 调整窗口大小

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 分离左右图
    left_img = frame[:, :half_width]
    right_img = frame[:, half_width:]

    # 转换为灰度图
    grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 计算视差
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_visual = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    disp_visual = cv2.applyColorMap(disp_visual.astype(np.uint8), cv2.COLORMAP_JET)

    # 重投影为3D点云
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # 简单检测鱼（用视差范围分割）
    mask = cv2.inRange(disparity, 10, 64)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        point = points_3D[cy, cx]
        Z = point[2]  # 深度值（mm）

        if Z <= 0 or Z > 10000:
            continue

        # 在左图上绘制检测结果
        cv2.rectangle(left_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            left_img,
            f"{Z/10:.1f} cm",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # 拼图显示：左图 + 伪彩视差
    combined = np.hstack((left_img, disp_visual))
    cv2.imshow("Stereo Depth", combined)

    # 写入输出视频（仅左图）
    out.write(left_img)

    # 检查按键输入
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # 按q退出
        break
    elif key == ord("p"):  # 按p暂停
        while True:
            key = cv2.waitKey(1)
            if key == ord("p") or key == ord("q"):
                break
        if key == ord("q"):
            break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"处理完成，结果已保存到 {output_path}")
