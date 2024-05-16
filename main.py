import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")

        self.N = 0
        self.points = None
        self.tetrahedron_indices = []

        self.create_widgets()

    def create_widgets(self):
        self.n_label = tk.Label(self.root, text="N (Number of points)")
        self.n_label.grid(row=0, column=0)
        self.n_entry = tk.Entry(self.root)
        self.n_entry.grid(row=0, column=1)

        self.generate_button = tk.Button(self.root, text="Generate Points", command=self.generate_points)
        self.generate_button.grid(row=1, column=0, columnspan=2)

        self.calculate_button = tk.Button(self.root, text="Calculate Tetrahedron", command=self.calculate_tetrahedron)
        self.calculate_button.grid(row=2, column=0, columnspan=2)

        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.grid(row=3, column=0, columnspan=2)

    def generate_points(self):
        try:
            self.N = int(self.n_entry.get())
            if self.N < 4:
                messagebox.showerror("Error", "Number of points must be at least 4!")
                return

            self.points = np.random.normal(size=(self.N, 3))
            self.result_text.delete(1.0, tk.END)  # Clear previous text
            self.result_text.insert(tk.END, "Generated Points:\n")
            for point in self.points:
                self.result_text.insert(tk.END, f"{point}\n")

            # Plot the points
            self.plot_points(self.points)
        except ValueError:
            messagebox.showerror("Error", "Invalid input!")

    def plot_points(self, points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        plt.show()

    def calculate_tetrahedron(self):
        if self.points is None or len(self.points) < 4:
            messagebox.showerror("Error", "Generate at least 4 points first!")
            return

        self.tetrahedron_indices = self.find_tetrahedron(self.points)
        self.result_text.insert(tk.END, "Tetrahedron Points:\n")
        for index in self.tetrahedron_indices:
            self.result_text.insert(tk.END, f"{self.points[index]}\n")

        # Plot the tetrahedron
        self.plot_tetrahedron(self.points, self.tetrahedron_indices)

    def find_tetrahedron(self, points):
        def distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        # Step 1: Find the most distant pair of points
        max_dist = 0
        point1, point2 = 0, 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = distance(points[i], points[j])
                if dist > max_dist:
                    max_dist = dist
                    point1, point2 = i, j

        # Step 2: Find the third point which is farthest from the line formed by point1 and point2
        max_dist = 0
        point3 = 0
        line_vec = points[point2] - points[point1]
        for i in range(len(points)):
            if i != point1 and i != point2:
                point_vec = points[i] - points[point1]
                dist = np.linalg.norm(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)
                if dist > max_dist:
                    max_dist = dist
                    point3 = i

        # Step 3: Find the fourth point which is farthest from the plane formed by point1, point2, and point3
        max_dist = 0
        point4 = 0
        normal_vec = np.cross(points[point2] - points[point1], points[point3] - points[point1])
        for i in range(len(points)):
            if i != point1 and i != point2 and i != point3:
                dist = abs(np.dot(points[i] - points[point1], normal_vec) / np.linalg.norm(normal_vec))
                if dist > max_dist:
                    max_dist = dist
                    point4 = i

        return [point1, point2, point3, point4]

    def plot_tetrahedron(self, points, tetrahedron_indices):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        tetra_points = points[tetrahedron_indices]
        ax.scatter(tetra_points[:, 0], tetra_points[:, 1], tetra_points[:, 2], c='r', marker='o')

        # Draw the edges of the tetrahedron
        for i in range(4):
            for j in range(i + 1, 4):
                ax.plot([tetra_points[i, 0], tetra_points[j, 0]],
                        [tetra_points[i, 1], tetra_points[j, 1]],
                        [tetra_points[i, 2], tetra_points[j, 2]], 'r-')

        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
