import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
from GeneticAlgorithm import genetic_algorithm as genetic_algorithm

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Preview with Genetic Algorithm")
        self.root.configure(bg='#f0f0f0')

        self.root.geometry("536x465")

        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12), padding=10, background='#4CAF50', foreground='white')
        self.style.configure('TLabel', font=('Helvetica', 12), background='#f0f0f0')
        self.style.configure('TFrame', background='#f0f0f0')

        self.frame = ttk.Frame(root, padding="10 10 10 10", style='TFrame')  
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.heading_label = ttk.Label(self.frame, text="Image Reconstruction", font=('Helvetica', 16, 'bold'), background='#f0f0f0')
        self.heading_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.upload_button = ttk.Button(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=1, column=0, columnspan=2, pady=20)

        self.original_image_label = ttk.Label(self.frame, text="Original Image", background='#f0f0f0')
        self.original_image_label.grid(row=2, column=0, pady=10)
        self.original_image_canvas = tk.Canvas(self.frame, width=256, height=256, bg='white')
        self.original_image_canvas.grid(row=3, column=0, pady=10)

        self.generated_image_label = ttk.Label(self.frame, text="Generated Image", background='#f0f0f0')
        self.generated_image_label.grid(row=2, column=1, pady=10)
        self.generated_image_canvas = tk.Canvas(self.frame, width=256, height=256, bg='white')
        self.generated_image_canvas.grid(row=3, column=1, pady=10)

        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)

        self.original_image = None
        self.generated_image = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.original_image_canvas, 256, 256)

            target_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            target_image = cv2.resize(target_image, (96, 96))
            self.run_genetic_algorithm(target_image)

    def run_genetic_algorithm(self, target_image):
        self.gen_algorithm = genetic_algorithm(target_image)
        self.update_generated_image()

    def update_generated_image(self):
        try:
            next_image = next(self.gen_algorithm)
            next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
            next_image = Image.fromarray(next_image)
            self.display_image(next_image, self.generated_image_canvas, 256, 256)
            self.root.after(100, self.update_generated_image)
        except StopIteration:
            pass

    def display_image(self, image, canvas, display_width, display_height):
        image = image.resize((display_width, display_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
