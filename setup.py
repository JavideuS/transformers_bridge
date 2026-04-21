from setuptools import find_packages, setup

package_name = 'transformers_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        ('share/' + package_name + '/launch',
            ['launch/default.launch.py', 'launch/fast.launch.py']),
        # Parameter files
        ('share/' + package_name + '/config',
            ['config/rtdetrv2.yaml']),

        # Install media files next to scripts so nodes can find them at runtime
        ('lib/' + package_name, ['madison.jpg']),
        ('lib/' + package_name, ['SukunaJogo_h264.mp4']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='javideus',
    maintainer_email='javi.rm2005@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detrv2 = transformers_bridge.detrv2:main',
            'test_image_pub = transformers_bridge.test_image_publisher:main',
            'venv_setup = scripts.venv_setup:main',
        ],
    },
)
