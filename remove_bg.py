import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

class InteractiveBackgroundRemover:
    def __init__(self):
        self.img = None
        self.original_img = None
        self.mask = None
        self.bgdModel = None
        self.fgdModel = None
        self.rect = None
        self.drawing = False
        self.mode = 'rect'  # 'rect', 'fg', 'bg'
        self.ix = -1
        self.iy = -1
        self.max_size = 800
        self.scale_factor = 1.0
        
        # 設定中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def load_image(self, img_path):
        """載入並預處理圖片"""
        print(f"📖 讀取圖片: {img_path}")
        self.original_img = cv2.imread(img_path)
        
        if self.original_img is None:
            print("❌ 無法讀取圖片!")
            return False
        
        print(f"✅ 圖片載入成功! 原始尺寸: {self.original_img.shape}")
        
        # 如果圖片太大，縮小處理
        h, w = self.original_img.shape[:2]
        if max(h, w) > self.max_size:
            self.scale_factor = self.max_size / max(h, w)
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.img = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"🔄 已縮放至: {self.img.shape} (縮放比例: {self.scale_factor:.2f})")
        else:
            self.img = self.original_img.copy()
            self.scale_factor = 1.0
        
        # 初始化遮罩和模型
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠回調函數"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                if self.mode == 'fg':
                    cv2.circle(self.mask, (x, y), 3, cv2.GC_FGD, -1)
                    cv2.circle(param, (x, y), 3, (0, 255, 0), -1)
                elif self.mode == 'bg':
                    cv2.circle(self.mask, (x, y), 3, cv2.GC_BGD, -1)
                    cv2.circle(param, (x, y), 3, (0, 0, 255), -1)
                    
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == 'rect':
                # 繪製矩形
                cv2.rectangle(param, (self.ix, self.iy), (x, y), (255, 0, 0), 2)
                self.rect = (min(self.ix, x), min(self.iy, y), 
                           abs(x - self.ix), abs(y - self.iy))
                print(f"✅ 已設定矩形區域: {self.rect}")

    def interactive_grabcut(self):
        """互動式 GrabCut 處理"""
        if self.img is None:
            print("❌ 請先載入圖片!")
            return False
        
        # 創建顯示視窗
        display_img = self.img.copy()
        cv2.namedWindow('互動式去背工具', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('互動式去背工具', 800, 600)
        cv2.setMouseCallback('互動式去背工具', self.mouse_callback, display_img)
        
        print("\n=== 操作說明 ===")
        print("1. 按 'r' 切換到矩形模式，拖拽選擇主要物件區域")
        print("2. 按 'f' 切換到前景模式，畫出要保留的區域（綠色）")
        print("3. 按 'b' 切換到背景模式，畫出要去除的區域（紅色）")
        print("4. 按 'g' 執行 GrabCut 算法")
        print("5. 按 'c' 清除所有標記")
        print("6. 按 's' 儲存結果")
        print("7. 按 'q' 或 ESC 退出")
        print("8. 按 'h' 重新顯示說明")
        
        while True:
            cv2.imshow('互動式去背工具', display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                self.mode = 'rect'
                print("📌 切換到矩形模式 - 拖拽選擇主要物件區域")
                
            elif key == ord('f'):
                self.mode = 'fg'
                print("📌 切換到前景模式 - 畫出要保留的區域（綠色）")
                
            elif key == ord('b'):
                self.mode = 'bg'
                print("📌 切換到背景模式 - 畫出要去除的區域（紅色）")
                
            elif key == ord('g'):
                if self.rect is not None:
                    print("🔄 執行 GrabCut 算法...")
                    self.run_grabcut()
                    self.show_result()
                    # 更新顯示
                    display_img = self.img.copy()
                    if self.rect:
                        cv2.rectangle(display_img, (self.rect[0], self.rect[1]), 
                                    (self.rect[0]+self.rect[2], self.rect[1]+self.rect[3]), 
                                    (255, 0, 0), 2)
                else:
                    print("⚠️ 請先用矩形模式選擇主要物件區域")
                    
            elif key == ord('c'):
                print("🔄 清除所有標記")
                self.mask = np.zeros(self.img.shape[:2], np.uint8)
                self.rect = None
                display_img = self.img.copy()
                
            elif key == ord('s'):
                if self.mask is not None:
                    self.save_result()
                else:
                    print("⚠️ 請先執行 GrabCut 算法")
                    
            elif key == ord('h'):
                print("\n=== 操作說明 ===")
                print("1. 按 'r' 切換到矩形模式，拖拽選擇主要物件區域")
                print("2. 按 'f' 切換到前景模式，畫出要保留的區域（綠色）")
                print("3. 按 'b' 切換到背景模式，畫出要去除的區域（紅色）")
                print("4. 按 'g' 執行 GrabCut 算法")
                print("5. 按 'c' 清除所有標記")
                print("6. 按 's' 儲存結果")
                print("7. 按 'q' 或 ESC 退出")
                print("8. 按 'h' 重新顯示說明")
                
            elif key == ord('q') or key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        return True

    def run_grabcut(self):
        """執行 GrabCut 算法"""
        try:
            if self.rect is not None:
                # 使用矩形初始化
                cv2.grabCut(self.img, self.mask, self.rect, 
                           self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            else:
                # 使用遮罩初始化
                cv2.grabCut(self.img, self.mask, None, 
                           self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            
            print("✅ GrabCut 算法執行完成")
            return True
            
        except Exception as e:
            print(f"❌ GrabCut 執行失敗: {e}")
            return False

    def show_result(self):
        """顯示處理結果"""
        if self.mask is None:
            return
        
        # 建立最終遮罩
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 255).astype('uint8')
        
        # 檢查前景比例
        foreground_pixels = np.sum(mask2 > 0)
        total_pixels = mask2.shape[0] * mask2.shape[1]
        fg_ratio = foreground_pixels / total_pixels * 100
        print(f"📊 前景比例: {fg_ratio:.1f}%")
        
        # 顯示結果比較
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原圖
        axes[0].imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("原始圖片")
        axes[0].axis('off')
        
        # 遮罩
        axes[1].imshow(mask2, cmap='gray')
        axes[1].set_title("生成的遮罩")
        axes[1].axis('off')
        
        # 結果
        result = self.img.copy()
        result[mask2 == 0] = [255, 255, 255]  # 白色背景
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title("去背結果")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

    def save_result(self):
        """儲存結果"""
        if self.mask is None:
            print("❌ 沒有可儲存的結果")
            return False
        
        # 如果有縮放，需要將遮罩放大回原始尺寸
        if self.scale_factor != 1.0:
            print("🔄 將遮罩放大回原始尺寸...")
            original_h, original_w = self.original_img.shape[:2]
            final_mask = np.where((self.mask == 2) | (self.mask == 0), 0, 255).astype('uint8')
            final_mask = cv2.resize(final_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            final_img = self.original_img
        else:
            final_mask = np.where((self.mask == 2) | (self.mask == 0), 0, 255).astype('uint8')
            final_img = self.img
        
        # 創建 RGBA 圖片
        b, g, r = cv2.split(final_img)
        rgba = cv2.merge((b, g, r, final_mask))
        
        # 儲存透明背景版本
        output_path = os.path.join(os.getcwd(), 'interactive_output_transparent.png')
        success = cv2.imwrite(output_path, rgba)
        
        if success:
            print(f"✅ 透明背景版本已儲存: {output_path}")
        
        # 儲存白色背景版本
        white_bg = final_img.copy()
        white_bg[final_mask == 0] = [255, 255, 255]
        white_output_path = os.path.join(os.getcwd(), 'interactive_output_white_bg.png')
        cv2.imwrite(white_output_path, white_bg)
        print(f"✅ 白色背景版本已儲存: {white_output_path}")
        
        return True

def select_image_file():
    """選擇圖片檔案"""
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    
    file_path = filedialog.askopenfilename(
        title="選擇要去背的圖片",
        filetypes=[
            ("圖片檔案", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("所有檔案", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

# 主程式
if __name__ == "__main__":
    print("=== 互動式 GrabCut 去背工具 ===")
    
    # 選擇圖片
    img_path = select_image_file()
    
    if not img_path:
        print("❌ 未選擇圖片，程式結束")
        exit()
    
    # 創建工具實例
    remover = InteractiveBackgroundRemover()
    
    # 載入圖片
    if remover.load_image(img_path):
        print("🎯 圖片載入成功，開始互動式處理...")
        
        # 開始互動式處理
        success = remover.interactive_grabcut()
        
        if success:
            print("\n🎉 處理完成!")
            print("💡 提示:")
            print("- 查看當前目錄下的 interactive_output_transparent.png (透明背景)")
            print("- 查看當前目錄下的 interactive_output_white_bg.png (白色背景)")
            print("- 如果效果不理想，可以重新執行並調整標記")
        else:
            print("\n❌ 處理失敗")
    else:
        print("❌ 圖片載入失敗")