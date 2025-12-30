import torch
import sys
import os

sys.path.append(os.getcwd())
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar

# --- CẤU HÌNH ---
# 1. Đường dẫn file SuGaR (.pt) đã fix lỗi
FIXED_CHECKPOINT = './output/coarse/garden/sugarcoarse_3Dgs2000_densityestim02_sdfnorm02/15000.pt'

# 2. Đường dẫn thư mục Vanilla 3DGS (Chứa file cameras.json)
# Lưu ý: Phải có dấu gạch chéo '/' ở cuối
VANILLA_DIR = './output/vanilla_gs/garden/'

class Args:
    # --- ĐƯỜNG DẪN QUAN TRỌNG ---
    # File model SuGaR (để lấy điểm, density)
    coarse_model_path = FIXED_CHECKPOINT
    
    # Thư mục gốc Vanilla (để lấy cameras.json)
    checkpoint_path = VANILLA_DIR
    
    scene_path = 'data/360_v2/garden'
    mesh_output_dir = './output/coarse_mesh/garden'
    
    # --- THAM SỐ TRÍCH XUẤT MESH ---
    surface_level = 0.3
    decimation_target = 200_000
    n_vertices_in_mesh = 200_000
    
    # --- THAM SỐ THUẬT TOÁN ---
    use_centers_to_extract_mesh = False
    use_marching_cubes = False
    poisson_depth = 10
    vertices_density_quantile = 0.1
    
    # --- CẤU HÌNH KHÔNG GIAN ---
    bboxmin = None
    bboxmax = None
    center_bbox = True
    project_mesh_on_surface_points = True
    use_vanilla_3dgs = False
    
    # --- CÁC THAM SỐ KHÁC ---
    square_size = 8
    gaussians_per_triangle = 6
    postprocess_mesh = False
    postprocess_density_threshold = 0.1
    postprocess_iterations = 5
    iteration = 15000
    iteration_to_load = 15000
    regularization_type = 'dn_consistency'
    gpu = 0
    eval = True
    white_background = False

args = Args()

print("Starting Mesh Extraction...")
print(f"Reading SuGaR model from: {args.coarse_model_path}")
print(f"Reading Cameras from: {args.checkpoint_path}")

try:
    extract_mesh_from_coarse_sugar(args)
    print("\n>>> SUCCESS! Mesh extraction complete.")
    print(f"Check your output folder: {args.mesh_output_dir}")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()