import os
from glob import glob
from setuptools import setup

package_name = 'whiteboard_eraser'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
        (os.path.join('share', package_name, 'models', 'materials', 'scripts'),
         glob('models/materials/scripts/*')),
        (os.path.join('share', package_name, 'models', 'materials', 'textures'),
         glob('models/materials/textures/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deng',
    maintainer_email='user@todo.todo',
    description='RM65 视觉擦白板仿真',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'detect_text_node = whiteboard_eraser.detect_text_node:main',
            'wipe_node = whiteboard_eraser.wipe_node:main',
            'gen_poem_texture = whiteboard_eraser.gen_poem_texture:main',
        ],
    },
)
