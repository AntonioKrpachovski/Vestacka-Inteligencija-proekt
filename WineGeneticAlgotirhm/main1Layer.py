import numpy as np
import pygad
import csvReader
import pygame
from neuralNetwork1Layer import forward_pass, fitness_func_factory, on_generation
import csv

with open("winequality.csv", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)

feature_names = header[:-1]

dataset = csvReader.read_file("winequality.csv")

X = np.array([[float(v) for v in row[:-1]] for row in dataset])
Y = np.array([1 if row[-1].strip().lower() == "good" else 0 for row in dataset])

input_size = X.shape[1]

min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
X_norm = (X - min_values) / (max_values - min_values + 1e-8) # za izbegnuvanje delenje so 0

#parametri na nevronska mrezha, da se menuvaat za eksperimentiranje
hidden_size = 5
output_size = 1
num_weights = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size

fitness_func = fitness_func_factory(X_norm, Y, input_size, hidden_size, output_size)

#generacii i broj na roditeli mozhat da se menuvaat za eksperimentiranje
ga_instance = pygad.GA(
    num_generations=500,
    num_parents_mating=25,
    fitness_func=fitness_func,
    sol_per_pop=200,
    num_genes=num_weights,
    mutation_percent_genes=15,
    mutation_type="random",
    crossover_type="single_point",
    #random_seed=42,
    keep_parents=5,
    on_generation=on_generation
)


ga_instance.run()


best_solution, best_fitness, _ = ga_instance.best_solution()
print("\nTraining finished.")
print("Best fitness (accuracy):", best_fitness)

#kod za vizuelizacija
pygame.init()
WIDTH, HEIGHT = 800, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wine Quality Predictor")

font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 18)

BG_COLOR = (245, 245, 250)
BUTTON_COLOR = (50, 200, 50)
BUTTON_HOVER = (30, 180, 30)
TEXT_COLOR = (0, 0, 0)

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, name):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = (min_val + max_val) / 2
        self.name = name
        self.dragging = False

    def draw(self, win):
        pygame.draw.rect(win, (200,200,200), (self.rect.x, self.rect.y + self.rect.height//2 - 5, self.rect.width, 10))

        t = (self.value - self.min_val) / (self.max_val - self.min_val + 1e-8)
        handle_x = self.rect.x + t * self.rect.width
        # boja na Slider
        color = (int(100*(1-t)), int(180*t), 255)
        pygame.draw.circle(win, color, (int(handle_x), self.rect.y + self.rect.height//2), 10)

        label = small_font.render(f"{self.name}: {self.value:.2f}", True, TEXT_COLOR)
        win.blit(label, (self.rect.x + self.rect.width + 20, self.rect.y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            rel_x = max(0, min(rel_x, self.rect.width))
            t = rel_x / self.rect.width
            self.value = self.min_val + t * (self.max_val - self.min_val)

sliders = [Slider(50, 50 + i*50, 300, 20, min_values[i], max_values[i], feature_names[i]) for i in range(input_size)]
button_rect = pygame.Rect(400, 750, 200, 40)
prediction_text = ""

#Loop za Pygame
running = True
while running:
    win.fill(BG_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        for slider in sliders:
            slider.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                values = np.array([s.value for s in sliders])
                arr = (values - min_values) / (max_values - min_values + 1e-8)
                output = forward_pass(best_solution, arr, input_size, hidden_size, output_size)
                prediction_text = "GOOD WINE" if output >= 0.5 else "BAD WINE"

    for slider in sliders:
        slider.draw(win)

    color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(win, color, button_rect)
    btn_text = font.render("Predict", True, (255,255,255))
    win.blit(btn_text, (button_rect.x + 40, button_rect.y + 5))

    pred_surface = font.render(prediction_text, True, (0,0,0))
    win.blit(pred_surface, (50, 700))

    pygame.display.update()

pygame.quit()
