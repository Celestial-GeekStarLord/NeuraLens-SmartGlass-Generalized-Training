import torch
from ultralytics import YOLO

def main():
    
    # RTX 3060 Ti has Ampere Tensor Cores; this enables faster 32-bit float math
    torch.backends.cudnn.benchmark = True 
    torch.cuda.empty_cache()

    print(f"--- Training Optimized for {torch.cuda.get_device_name(0)} ---")

    
    # CONFIGURATION5
    
    DATA_YAML = r"C:\Users\Lucas Dev\Downloads\neuralens_generalized\data.yaml"
    
    # Since you have 8GB VRAM, we can use the 'Medium' model (yolo11m.pt) 
    # for significantly better generalization than the 'Nano' model.
    MODEL_WEIGHTS = r"C:\Users\Lucas Dev\Downloads\neuralens_generalized\runs\detect\train4\weights\best.pt"
    
    IMG_SIZE = 640 
    BATCH_SIZE = 16
    EPOCHS = 150     
    DEVICE = 0       

    # Load Model
    model = YOLO(MODEL_WEIGHTS)

    
    # TRAINING SETTINGS
    
    model.train(
        data=DATA_YAML,
        resume=True,       # Start fresh training (not resuming)
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        # 'workers' speeds up data loading. For 3060 Ti on Windows, 4-8 is ideal.
        workers=8, 
        patience = 30,
        # --- GPU & SENSOR CORE TECH ---
        amp=True,           # Enables Automatic Mixed Precision (Uses Tensor Cores)
        plots=True,         # Generates training charts
        
        # --- GENERALIZATION & ACCURACY ---
        optimizer="AdamW",  # Better for modern RTX cards than SGD
        lr0=0.0003,
        lrf=0.01,          # Initial learning rate for AdamW
        cos_lr=True,        # Use cosine learning rate scheduler for better convergence
        
        # --- AUGMENTATION (Generalization) ---
        mosaic=1.0,         # Combines 4 images to help detect small objects
        mixup=0.2,          # Creates composite images to prevent overfitting
        copy_paste=0.2,     # Good for segmenting/detecting objects in cluttered scenes
        degrees=10.0,  
        
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        scale=0.7,
        translate=0.2,     # Slight rotations for robustness
    )

if __name__ == "__main__":
    main()
    print("✅ Training Complete!")