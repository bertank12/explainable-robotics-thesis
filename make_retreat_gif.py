from PIL import Image
import os

# Kaynak klasör ve gif ismi
frame_dir = "retreat_frames"
output_gif = "retreat_simulation.gif"

# Tüm .png frame'leri sıraya koy
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

# İlk frame'i aç
frames = [Image.open(os.path.join(frame_dir, f)) for f in frame_files]

# GIF oluştur
frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    duration=60,      # her frame süresi (ms)
    loop=0
)

print(f"GIF başarıyla oluşturuldu: {output_gif}")

