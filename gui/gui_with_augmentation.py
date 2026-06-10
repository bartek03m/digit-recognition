import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie Cyfr")
        self.root.resizable(False, False)
        
        try:
            self.model = tf.keras.models.load_model('models/digit-recognizer-with-augmentation.keras')
            print("Model załadowany pomyślnie!")
        except Exception as e:
            print(f"Błąd podczas ładowania modelu: {e}")
            self.model = None

        self.canvas_width = 280
        self.canvas_height = 280
        self.brush_size = 13
        
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black', cursor='cross')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw_engine = ImageDraw.Draw(self.image)

        self.btn_predict = tk.Button(self.root, text="Rozpoznaj cyfrę", font=("Arial", 12, "bold"), command=self.predict_digit)
        self.btn_predict.grid(row=1, column=0, pady=5, sticky="ew", padx=10)

        self.btn_clear = tk.Button(self.root, text="Wyczyść", font=("Arial", 12), command=self.clear_canvas)
        self.btn_clear.grid(row=1, column=1, pady=5, sticky="ew", padx=10)

        self.label_result = tk.Label(self.root, text="Narysuj cyfrę na czarnym polu!", font=("Arial", 14))
        self.label_result.grid(row=2, column=0, columnspan=2, pady=15)

    def draw(self, event):
        x, y = event.x, event.y
        r = self.brush_size
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
        self.draw_engine.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_engine.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.label_result.config(text="Narysuj cyfrę na czarnym polu!")

    def predict_digit(self):
        if self.model is None:
            self.label_result.config(text="Błąd: Brak modelu!")
            return
        
        bbox = self.image.getbbox()
        if bbox is None:
            self.label_result.config(text="Najpierw coś narysuj!")
            return

        cropped = self.image.crop(bbox)

        # Rescaling with MINST standard (width=20px)
        width, height = cropped.size
        ratio = 20.0 / max(width, height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        resized = cropped.resize(new_size, resample=Image.Resampling.LANCZOS)

        final_image = Image.new("L", (28, 28), color=0)
        offset_x = (28 - new_size[0]) // 2
        offset_y = (28 - new_size[1]) // 2
        final_image.paste(resized, (offset_x, offset_y))
        
        # final_image.save("debug.png")

        img_array = np.array(final_image)
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32')

        prediction = self.model.predict(img_array, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        self.label_result.config(text=f"Myślę, że to: {digit} (pewność: {confidence:.1f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()