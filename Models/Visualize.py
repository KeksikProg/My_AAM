import matplotlib.pyplot as plt

def plot_model_shape(flat_model, title, target_flat=None):
     model = flat_model.reshape(-1, 2)
     plt.plot(model[:, 0], model[:, 1], 'o-', label="Собранная модель")
     if target_flat is not None:
         target_model = target_flat.reshape(-1, 2)
         plt.plot(target_model[:, 0], target_model[:, 1], 'x--', label="Целевая модель")
     plt.gca().invert_yaxis()
     plt.title(title)
     plt.legend()
     plt.axis('equal')
     plt.grid(True)
     plt.show()

def show_texture(texture):
        plt.imshow(texture, cmap='gray')
        plt.axis('off')
        plt.show()