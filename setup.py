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
        ],
    },
)
