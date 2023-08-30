import streamlines as sl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# create a 2D vector field
N = 100
x_range = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x_range, x_range)
R = np.sqrt(X**2 + Y**2)
dy, dx = np.gradient(R)

# create structure tensor
T = np.zeros((dx.shape[0], dy.shape[0], 2, 2))
T[:, :, 0, 0] = dy * dy
T[:, :, 1, 1] = dx * dx
T[:, :, 0, 1] = dx * dy
T[:, :, 1, 0] = T[:, :, 0, 1]

# calculate the eigenvectors
vec_field = np.empty((T.shape[0], T.shape[1], 2))
eigvals, eigvecs = np.linalg.eigh(T)
vec_field = eigvecs[:, :, 1]


#%%
# Create a streamplot
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, dx, dy, density=1.5, linewidth=1, cmap='viridis')

# Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Streamplot Example')

# Show the plot
plt.show()

#%%
# create random points on a circle
'''num_points = 50

angles = np.random.uniform(0, 2 * np.pi, num_points)

radius = 5  # Adjust this radius as needed
x_coordinates = radius * np.cos(angles)
y_coordinates = radius * np.sin(angles)

seed_pts = []
for x, y in zip(x_coordinates, y_coordinates):
    if -5 <= x <= 5 and -5 <= y <= 5:
        seed_pts.append((x, y))'''


#%%
# create random points on the plane
num_points = 50
seed_pts = [(np.random.uniform(-8.0, 8.0), np.random.uniform(-8.0, 8.0)) for _ in range(num_points)]

#%%
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

major = np.arange(-10, 101, 10)
minor = np.arange(0, 101, 5)
ax.set_xticks(major)
ax.set_xticks(minor, minor=True)
ax.set_yticks(major)
ax.set_yticks(minor, minor=True)

plt.imshow(np.zeros_like(X), extent=(-10, 10, -10, 10), cmap='gray')
plt.scatter(*zip(*seed_pts), color='red', label='Seed Points')
plt.legend()
plt.grid()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
#plt.show()

#%%
img_range = [[-10, 10], [-10, 10]]
all_lines = sl.vec2streamline_2d(vec_field, seed_pts, img_range)

for i in range(num_points):
    plt.scatter(*zip(*all_lines[i]), color='blue', marker='.', label='Seed Points')
    plt.show()
#%%
'''point = [(5, 5)]
line = vec2streamline_2d(vec_field, point, img_range)
plt.plot(point[0][0], point[0][1], color='red', marker='o')
plt.scatter(*zip(*line[0]), color='blue', marker='.', label='Seed Points')'''





