import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

def view3D( data3D, filename='_.gif', path='./images/' ):

    # set seaborn darkgrid theme
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots()

    def animate(frame_num):
        ax.clear()
        ax.imshow(abs(data3D[frame_num,:,:]), cmap='gray')
        return ax

    anim = FuncAnimation(fig, animate, frames=data3D.shape[2], interval=1)

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    path = os.path.join(path, filename)
    print(path)
    anim.save(path)