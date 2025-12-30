import torch
import os

# Đường dẫn file lỗi
input_path = './output/coarse/garden/sugarcoarse_3Dgs2000_densityestim02_sdfnorm02/15000.pt'
# Đường dẫn file mới sau khi sửa
output_path = './output/coarse/garden/sugarcoarse_3Dgs2000_densityestim02_sdfnorm02/15000_fixed.pt'

print(f"Loading: {input_path}")
checkpoint = torch.load(input_path, map_location="cuda:0")
state_dict = checkpoint['state_dict']

# 1. Lấy số lượng điểm
num_points = state_dict['_points'].shape[0]
print(f"Number of points: {num_points}")

# 2. Tạo Opacity giả lập (nếu thiếu)
if '_opacities' not in state_dict:
    print("Fixing missing '_opacities'...")
    # Tạo opacity = 1.0 (đậm đặc) cho toàn bộ điểm
    ones_opacity = torch.ones((num_points, 1), dtype=torch.float, device="cuda:0")
    # Quan trọng: Cần dùng hàm inverse sigmoid vì model thường lưu logit
    # Tuy nhiên để an toàn, ta cứ lưu raw 1.0, hàm activation sẽ xử lý sau
    state_dict['_opacities'] = ones_opacity
else:
    print("'_opacities' already exists.")

# 3. Lưu lại file mới
torch.save(checkpoint, output_path)
print(f"Saved fixed checkpoint to: {output_path}")