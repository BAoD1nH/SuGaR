import torch

# Đường dẫn file checkpoint của bạn
ckpt_path = './output/coarse/garden/sugarcoarse_3Dgs2000_densityestim02_sdfnorm02/15000.pt'

try:
    checkpoint = torch.load(ckpt_path, map_location="cuda:0")
    print("\n=== DANH SÁCH CÁC KEY TRONG FILE CHECKPOINT ===")
    print(checkpoint.keys())
    print("===============================================\n")
    
    # Kiểm tra thử một vài key phổ biến
    if 'state_dict' in checkpoint:
        print("Phát hiện lồng nhau! Key thực sự nằm trong 'state_dict'.")
        print("Keys con:", checkpoint['state_dict'].keys())
        
except Exception as e:
    print("Lỗi khi đọc file:", e)