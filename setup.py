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
            ['launch/default.launch.py', 'launch/fast.launch.py',
             'launch/compare.launch.py']),
        # Parameter files
        ('share/' + package_name + '/config',
            ['config/default.yaml', 'config/yolo.yaml', 'config/custom.yaml']),
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
            'detector = transformers_bridge.detector_node:main',
            'test_image_pub = transformers_bridge.test_image_publisher:main',
            'venv_setup = scripts.venv_setup:main',
            'list_models = transformers_bridge.model_registry:main',
        ],
    },
)
