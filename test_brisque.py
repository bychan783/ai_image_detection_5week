import os
import numpy as np
from PIL import Image
from brisque import BRISQUE

# 1. 평가할 이미지 6장 목록 (파일명, 카테고리명)
image_list = [
    ("img1.jpg", "img_normal.jpg"),                      # 일반 사각형 이미지
    ("img2.png", "img_ai.png"),                          # AI로 만든 사각형 이미지
    ("img3.jpg", "img_cartoon.jpg"),                     # 만화로 만든 흑백 스케치
    ("img4.png", "img_ai_sketch_black.png"),             # AI로 만든 흑백 스케치
    ("img5.jpg", "img_sketch.jpg"),                      # AI 없이 만든 흑백 스케치
    ("img6.png", "img_ai_sketch_color.png")              # AI로 만든 컬러 스케치
]

def main():
    obj = BRISQUE(url=False)
    img_num = 1 

    # 표 상단 헤더 출력
    print("=" * 72)
    print(f"{'번호':<5} | {'파일명':<23} | {'카테고리':<23} | {'BRISQUE 점수'}")
    print("-" * 72)

    # 2. 이미지별 점수 계산 및 표 형식 출력
    for filename, label in image_list:
       
        if os.path.exists(filename):
            try:
                img = Image.open(filename).convert('RGB')
                ndarray = np.asarray(img)
                raw_score = obj.score(img=ndarray)
                
                # 0 ~ 100 사이로 점수 범위 제한 (0 미만은 0으로, 100 초과는 100으로 고정)
                score_val = max(0.0, min(100.0, float(raw_score)))
                score_str = f"{score_val:>8.2f}점"
            except Exception:
                score_str = "    에러"
        else:
            score_str = "  파일없음"
        
        print(f"{img_num:<5} | {filename:<23} | {label:<23} | {score_str}")
        img_num += 1
    # 표 하단 마감
    print("=" * 72)

if __name__ == "__main__":
    main()
