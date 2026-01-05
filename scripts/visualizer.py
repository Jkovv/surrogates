import os
import pyvista as pv
import matplotlib.pyplot as plt

EXPORT_ROOT = "/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/cc3d_export_s42"
MODELS = ["BASELINE", "PINN", "DEEPONET", "PI_DEEPONET", "GPR", "STA_LSTM"]
GRIDS = [50, 100, 250, 500]
WINDOWS = ["Window_82_100", "Window_72_89"]
CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]

def generate_images():
    for res in GRIDS:
        for model in MODELS:
            for window in WINDOWS:
                path = os.path.join(EXPORT_ROOT, model, f"{res}x{res}", window)
                if not os.path.exists(path): continue
                
                output_dir = f"visualizations/{model}_{res}_{window}"
                os.makedirs(output_dir, exist_ok=True)
                
                files = sorted([f for f in os.listdir(path) if f.endswith(".vtk")])
                for f in files:
                    grid = pv.read(os.path.join(path, f))
                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                    
                    for i, name in enumerate(CYTOKINES):
                        ax = axes[i//3, i%3]
                        if name in grid.point_data:
                            data = grid.point_data[name].reshape((res, res), order="F")
                            ax.imshow(data, cmap='viridis', origin='lower')
                            ax.set_title(name)
                        ax.axis('off')
                    
                    plt.savefig(os.path.join(output_dir, f.replace(".vtk", ".png")))
                    plt.close()
                print(f"Finished: {model} {res}x{res} {window}")

if __name__ == "__main__":
    generate_images()
