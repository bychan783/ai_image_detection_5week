import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_images(img_size=512):
    """4가지 원본 실습 이미지를 생성하여 리스트로 반환하는 함수"""
    center = img_size // 2
    
    # 1. 가운데 점 1개 (원 형태)
    img_center = np.zeros((img_size, img_size), dtype=np.uint8)
    img_center[center, center] = 255

    # 2. 정사각형 꼭짓점 4개
    img_square = np.zeros((img_size, img_size), dtype=np.uint8)
    d_sq = img_size //8
    sqDot_list = [
        (center - d_sq, center - d_sq),
        (center - d_sq, center + d_sq),
        (center + d_sq, center - d_sq),
        (center + d_sq, center + d_sq),
    ]
    for y, x in sqDot_list:
        img_square[x, y] = 255

    # 3. 직사각형 꼭짓점 4개
    img_rect = np.zeros((img_size, img_size), dtype=np.uint8)
    dx, dy = img_size //8, img_size //8*3
    rectDot_list = [
        (center - dy, center - dx),
        (center - dy, center + dx),
        (center + dy, center - dx),
        (center + dy, center + dx),
    ]
    for y, x in rectDot_list:
        img_rect[x, y] = 255
        

    # 4. 사인 그래프 (주기 = PI)
    img_sin = np.zeros((img_size, img_size), dtype=np.float32)
    x_space = np.linspace(0, 4 * np.pi, img_size)
    sine_wave_255 = (np.sin(2 * x_space) + 1) * 127.5
    for i in range(img_size):
        img_sin[:, i] = sine_wave_255[i]

    return [img_center, img_square, img_rect, img_sin]


def perform_fft(images):
    """
    이미지 리스트를 입력받아 2D 푸리에 변환(FFT)을 수행하고,
    1) 시각화(화면 출력)를 위한 스펙트럼 이미지 리스트(fft_images)와
    2) 원본 복원(IFFT)을 위한 복소수 데이터 리스트(dft_shift_list)를 반환하는 함수
    """
    
 
    fft_images = []      # 사람 눈으로 보기 위해 로그 스케일로 압축된 스펙트럼 (화면 출력용)
    dft_shift_list = []  # 수학적 복원(IFFT)을 위해 위상 정보와 실수/허수를 모두 담은 데이터 (계산 저장용)

    for img in images:
        
        # --- (A) 푸리에 변환을 위한 데이터 타입 변경 ---
        # OpenCV의 dft() 함수는 float32(32비트 실수형) 타입만 입력으로 받으므로 형변환을 해줍니다.
        img_float = np.float32(img)
        
        # --- (B) 2D 이산 푸리에 변환 (DFT: Discrete Fourier Transform) 수행 ---
        # flags=cv2.DFT_COMPLEX_OUTPUT: 변환 결과를 '실수부(Real)'와 '허수부(Imaginary)' 
        # 2개의 채널을 가진 3차원 배열(H, W, 2) 형태로 출력하라는 옵션입니다.
        dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # --- (C) 저주파 영역을 이미지 정중앙으로 이동 (Shift) ---
        # 기본 FFT 결과는 저주파(DC 성분, 이미지의 전반적인 밝기/배경 정보)가 좌상단(0,0)에 모여 있습니다.
        # 시각적 분석과 처리를 쉽게 하기 위해 fftshift()를 써서 저주파를 이미지 '정중앙'으로 옮겨줍니다.
        dft_shift = np.fft.fftshift(dft)
        
        # ★ [중요] IFFT(3단계)에서 원본으로 손실 없이 100% 복원하기 위해 
        # 위상(Phase) 정보가 포함된 날것(Raw)의 dft_shift 데이터를 리스트에 보관합니다.
        dft_shift_list.append(dft_shift)

        # --- (D) 시각화를 위한 주파수 크기(Magnitude) 계산 ---
        # dft_shift[:, :, 0]은 실수부(Real), dft_shift[:, :, 1]은 허수부(Imag)입니다.
        # cv2.magnitude(a, b)는 sqrt(a^2 + b^2) 공식을 계산하여 주파수의 강도(크기)를 구합니다.
        # (참고: 이 과정을 거치면서 '위상 정보'는 제거됩니다.)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        
        # --- (E) 화면에 출력하기 위한 로그 스케일(Log Scale) 압축 ---
        # 푸리에 변환 직후의 magnitude 값은 중앙(저주파) 수치가 수백만 단위로 매우 크기 때문에,
        # 그냥 출력하면 중앙 점만 하얗고 나머진 전부 시각적으로 새까맣게 보입니다.
        # 따라서 20 * log(...) 연산을 통해 큰 값은 압축하고 작은 값은 키워 사람 눈에 패턴이 보이게 만듭니다.
        # (+1을 더하는 이유: log(0)은 수학적으로 정의되지 않아 에러가 나므로 0을 방지하기 위함)
        magnitude_spectrum = 20 * np.log(magnitude + 1)
        
        # 시각화용으로 가공이 완료된 스펙트럼 이미지를 화면 출력용 리스트에 보관합니다.
        fft_images.append(magnitude_spectrum)

    # [3. 두 가지 리스트를 튜플 형태로 동시에 반환]
    # (시각화용 이미지 리스트, 역변환 계산용 복소수 데이터 리스트)
    return fft_images, dft_shift_list

def perform_ifft(dft_shift_list):
    """
    FFT 과정에서 생성된 주파수 영역 데이터(dft_shift_list)를 입력받아
    2D 역 푸리에 변환(IFFT)을 수행하고, 원본 이미지로 복원하여 반환하는 함수
    """

    ifft_images = []


    for dft_shift in dft_shift_list:
        
        # --- (A) 중앙으로 모았던 저주파(DC 성분)를 원래 위치로 되돌림 ---
        # FFT 단계에서 분석을 편하게 하려고 fftshift()를 써서 저주파를 화면 정중앙으로 옮겼었습니다.
        # 역변환(idft)을 수행하려면 저주파가 다시 원래 위치인 '좌상단(0, 0)'에 있어야 하므로
        # np.fft.ifftshift()를 사용해 역으로 이동시켜 줍니다.
        f_ishift = np.fft.ifftshift(dft_shift)
        
        # --- (B) 2D 역 이산 푸리에 변환 (IDFT: Inverse DFT) 수행 ---
        # 주파수 영역의 데이터를 다시 우리가 눈으로 보는 '공간 영역(Pixel Image)'으로 되돌립니다.
        # ★ [핵심 플래그] flags=cv2.DFT_SCALE
        #   - OpenCV의 dft/idft는 연산 후 결과값이 전체 픽셀 수(512x512)만큼 뻥튀기(비례 증가)됩니다.
        #   - DFT_SCALE 플래그를 넣어주어야 뻥튀기된 값을 전체 크기로 나누어(1/N)
        #     원래 이미지의 픽셀 스케일(0~255 범위 등)로 정확하게 정상화시켜 줍니다.
        #   - (이 옵션이 없으면 float32 타입인 사인 웨이브 이미지가 하얗게 날아가 버립니다!)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
        
        # --- (C) 실수부와 허수부 2채널을 1채널 크기(Magnitude) 이미지로 변환 ---
        # img_back[:, :, 0]은 실수부(Real), img_back[:, :, 1]은 허수부(Imag)입니다.
        # 역변환이 완료되었더라도 연산 상의 미세한 부동소수점 오차로 허수부 값이 아주 작게 남아있을 수 있으므로,
        # cv2.magnitude()를 사용해 sqrt(Real^2 + Imag^2) 공식을 거쳐 완벽한 1채널 실수 이미지로 만듭니다.
        img_restored = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        # 완벽하게 원본 상태로 복원된 이미지를 리스트에 추가합니다.
        ifft_images.append(img_restored)

    # [3. 복원된 이미지 4개가 담긴 리스트 반환]
    return ifft_images

def show_all_results_in_one_figure(images, fft_images, ifft_images):
    """원본, FFT, IFFT 결과를 3행 4열의 하나의 창에 모두 시각화하는 함수"""
    col_titles = ["1. Center Dot", "2. Square", "3. Rectangle", "4. Sine Wave"]
    row_labels = ["Original", "2D - FFT", "2D - IFFT"]

    # 3행 4열 하나의 창 생성 (가로 16, 세로 11 비율)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))

    for col in range(4):
        # 1행: 원본 이미지
        axes[0, col].imshow(images[col], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"[{row_labels[0]}] {col_titles[col]}", fontsize=11)
        axes[0, col].axis("off")

        # 2행: 2D FFT 변환 결과
        axes[1, col].imshow(fft_images[col], cmap="gray")
        axes[1, col].set_title(f"[{row_labels[1]}] {col_titles[col]}", fontsize=11)
        axes[1, col].axis("off")

        # 3행: 2D IFFT 복원 결과
        axes[2, col].imshow(ifft_images[col], cmap="gray", vmin=0, vmax=255)
        axes[2, col].set_title(f"[{row_labels[2]}] {col_titles[col]}", fontsize=11)
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================
# 메인 실행부 (함수 순차 호출)
# ==========================================
if __name__ == "__main__":
    # 1. 원본 이미지 4개 생성
    original_images = create_images(img_size=64)

    # 2. 2D FFT 변환 수행
    fft_images, dft_shift_list = perform_fft(original_images)

    # 3. 2D IFFT 복원 수행
    ifft_images = perform_ifft(dft_shift_list)

    # 4. 단 하나의 창에서 전체 12개 이미지 시각화
    show_all_results_in_one_figure(original_images, fft_images, ifft_images)
