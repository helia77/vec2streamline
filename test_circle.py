import streamlines as sl
import numpy as np
import matplotlib.pyplot as plt

# create a 2D vector field
N = 100
M = 200
range_0 = np.linspace(-10, 10, N)
range_1 = np.linspace(-20, 20, M)
mesh = np.meshgrid(range_0, range_1)
R = np.sqrt(mesh[0]**2 + mesh[1]**2)
d0 = np.gradient(R, axis=0)
d1 = np.gradient(R, axis=1)

# create structure tensor
T = np.zeros((d0.shape[0], d1.shape[1], 2, 2))
T[:, :, 0, 0] = d0 * d0
T[:, :, 1, 1] = d1 * d1
T[:, :, 0, 1] = d0 * d1
T[:, :, 1, 0] = T[:, :, 0, 1]

all_lines, seeds = sl.tensor2streamlines(T, 8)

#for li in range(len(all_lines)):
#    plt.plot(all_lines[li][:, 0], all_lines[li][:, 1])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
#plt.imshow(np.zeros_like(X), extent=(0, 100, 0, 100), cmap='gray')
#plt.imshow(R)
for i in range(len(all_lines)):
    #plt.scatter(*zip(*all_lines[i]), color='blue', marker='.', label='Seed Points')
    plt.plot(all_lines[i][:, 0], all_lines[i][:, 1])
    plt.show()
plt.scatter(*zip(*seeds), color='red', label='Seed Points')

#%%
# Create a streamplot
plt.figure(figsize=(8, 6))
plt.streamplot(mesh[0], mesh[1], d1, d0, linewidth=1.5)

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
all_lines = sl.vec2streamlines(T, seed_pts, img_range)

for i in range(len(all_lines)):
    plt.scatter(*zip(*all_lines[i]), color='blue', marker='.', label='Seed Points')
    plt.show()
#%%
'''point = [(5, 5)]
line = vec2streamline_2d(vec_field, point, img_range)
plt.plot(point[0][0], point[0][1], color='red', marker='o')
plt.scatter(*zip(*line[0]), color='blue', marker='.', label='Seed Points')'''





