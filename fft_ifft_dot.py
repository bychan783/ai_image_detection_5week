import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_images(img_size=64):
  
    center = img_size // 2

    # 1. 가운데 점 1개 (딱 1픽셀)
    img_center = np.zeros((img_size, img_size), dtype=np.uint8)
    img_center[center, center] = 255

    # 2. 정사각형 꼭짓점 4개 (딱 1픽셀씩)
    img_square = np.zeros((img_size, img_size), dtype=np.uint8)
    d_sq = img_size // 8  # 중심에서 ±8 픽셀 거리
    sqDot_list = [
        (center - d_sq, center - d_sq),
        (center - d_sq, center + d_sq),
        (center + d_sq, center - d_sq),
        (center + d_sq, center + d_sq),
    ]
    for y, x in sqDot_list:
        img_square[y, x] = 255

    # 3. 직사각형 꼭짓점 4개 (딱 1픽셀씩, 세로는 짧고 가로는 넓게)
    img_rect = np.zeros((img_size, img_size), dtype=np.uint8)
    dy, dx = img_size // 8, img_size // 4  # 세로 ±8, 가로 ±16 거리
    rectDot_list = [
        (center - dy, center - dx),
        (center - dy, center + dx),
        (center + dy, center - dx),
        (center + dy, center + dx),
    ]
    for y, x in rectDot_list:
        img_rect[y, x] = 255

    # 4. 사인 그래프 (주기 = PI)
    img_sin = np.zeros((img_size, img_size), dtype=np.float32)
    x_space = np.linspace(0, 4 * np.pi, img_size)
    sine_wave_255 = (np.sin(2 * x_space) + 1) * 127.5
    for i in range(img_size):
        img_sin[:, i] = sine_wave_255[i]

    return [img_center, img_square, img_rect, img_sin]


def perform_fft(images):
    fft_images = []
    dft_shift_list = []

    for img in images:
        img_float = np.float32(img)
        dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        dft_shift_list.append(dft_shift)

        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude_spectrum = 20 * np.log(magnitude + 1)
        fft_images.append(magnitude_spectrum)

    return fft_images, dft_shift_list


def perform_ifft(dft_shift_list):
    ifft_images = []

    for dft_shift in dft_shift_list:
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
        img_restored = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        ifft_images.append(img_restored)

    return ifft_images


def show_all_results_in_one_figure(images, fft_images, ifft_images):
    col_titles = ["1. Center Dot", "2. Square", "3. Rectangle", "4. Sine Wave"]
    row_labels = ["Original", "2D - FFT", "2D - IFFT"]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    for col in range(4):
        # 원본 (1픽셀이 잘 보이도록 vmin=0, vmax=255 고정)
        axes[0, col].imshow(images[col], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"[{row_labels[0]}] {col_titles[col]}", fontsize=11)
        axes[0, col].axis("off")

        # 2D FFT (격자선이 또렷하게 보이도록 자동 콘트라스트)
        axes[1, col].imshow(fft_images[col], cmap="gray")
        axes[1, col].set_title(f"[{row_labels[1]}] {col_titles[col]}", fontsize=11)
        axes[1, col].axis("off")

        # 2D IFFT 복원
        axes[2, col].imshow(ifft_images[col], cmap="gray", vmin=0, vmax=255)
        axes[2, col].set_title(f"[{row_labels[2]}] {col_titles[col]}", fontsize=11)
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 64x64 규격으로 실행
    original_images = create_images(img_size=64)
    fft_images, dft_shift_list = perform_fft(original_images)
    ifft_images = perform_ifft(dft_shift_list)
    show_all_results_in_one_figure(original_images, fft_images, ifft_images)
