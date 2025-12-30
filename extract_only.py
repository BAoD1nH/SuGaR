import torch
import os
from sugar_scene.sugar_model import SuGaR
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar

# --- CẤU HÌNH ---
# Đường dẫn checkpoint (Đúng như bạn đã cung cấp)
checkpoint_path = './output/coarse/garden/sugarcoarse_3Dgs2000_densityestim02_sdfnorm02/15000.pt'
scene_path = 'data/360_v2/garden'
output_dir = './output/coarse_mesh/garden'

class Args:
    scene_path = scene_path
    checkpoint_path = checkpoint_path
    mesh_output_dir = output_dir
    surface_level = 0.3
    n_vertices_in_mesh = 200_000  # Low poly config
    bboxmin = None
    bboxmax = None
    center_bbox = True
    project_mesh_on_surface_points = True
    use_vanilla_3dgs = False
    
    # Các tham số giả lập
    iteration = 15000
    regularization_type = 'dn_consistency'
    gpu = 0

args = Args()

print(f"Loading checkpoint from: {args.checkpoint_path}")
checkpoint = torch.load(args.checkpoint_path, map_location="cuda:0")

# Lấy state_dict ra trước
state_dict = checkpoint['state_dict']

# Xử lý Opacities (Vì file checkpoint của bạn thiếu key này)
# Chúng ta sẽ tạo opacities mặc định là 1.0 (đặc) cho tất cả các điểm
num_points = state_dict['_points'].shape[0]
print(f"Number of points found: {num_points}")
all_opacities = torch.ones((num_points, 1), dtype=torch.float, device="cuda:0")

# Khởi tạo model SuGaR
# Lưu ý: Truy cập vào state_dict thay vì checkpoint trực tiếp
sugar = SuGaR(
    all_points=state_dict['_points'],
    all_densities=state_dict['all_densities'],
    all_sh_coordinates_dc=state_dict['_sh_coordinates_dc'],
    all_sh_coordinates_rest=state_dict['_sh_coordinates_rest'],
    all_scales=state_dict['_scales'],
    all_quaternions=state_dict['_quaternions'],
    all_opacities=all_opacities, # Dùng opacity giả lập
    triangle_scale=1.0,
    surface_level=args.surface_level
)

print("Model loaded successfully!")
print(f"Extracting Mesh to {args.mesh_output_dir}...")
extract_mesh_from_coarse_sugar(args)
print(">>> DONE! Hãy kiểm tra folder output.")