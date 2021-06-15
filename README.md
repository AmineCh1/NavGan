# NavGan
Semester Project : Navigating in GAN's latent spaces.

## For images and toy_datasets: 
In each folder, you will find a notebook containing a quick showcase of the implemented functions. Furthermore, each function has been commented.
### Libraries used: 
- Python 3.7.9
- Pytorch 1.7.1
- Numpy 1.19.2
- Matplotlib 3.3.2 
- Scipy 1.1.0

In case of reproducibility difficulties, I have added the  yml file to reproduce the environment.

## For WarpGAN: 

### Cloning repository : 
https://github.com/seasonSH/WarpGAN

### Downloading pretrained model: 
https://drive.google.com/file/d/1XwjMGcYIg2qwEKHsC7uSmZayHvnEFhyg/view

### 
In WarpGAN/, copy (and replace) ```test.py```, and include ```extract_style_vectors.py``` and ```extract_warp.py```. 
Include downloaded model in WarpGAN/pretrained
### Running `test.py`: 
Args:
    - ```model_dir```: Path of pretrained model, usually in `WarpGAN/pretrained`.
    - ```input```: Path of input image. Usually in `WarpGAN/data/example`.
    - ```output```: Path to store output, usually in `WarpGAN/result`.
    - ```num_styles```: Number of different styles to generate.
    - ```scale```: Scale of geometric deformation.
    - ```nb_dir```: Number of eigen directions to vary on. 

    
### Librairies used
- Python 3.7.9
- Tensorflow 1.13.0rc1
- Scipy 1.2.1
- Numpy 1.16.1

I have also added a yml file to easily reproduce his environment. 


I have written the code myself for the image and toy dataset visual- ```model_dir```: Path of pretrained model, usually in `WarpGAN/pretrained`.izations. 
For WarpGAN: 
For VAE models: 