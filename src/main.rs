extern crate rand;
mod model;
mod loadmnist;
use crate::model::layer::util::*;

fn main() {
   let num_train_images = 10000;
   let num_test_image = 10000;
   let image_width = 28;
   let image_height = 28;

   let train_images = loadmnist::read_mnist_images(true);
   let train_labels = loadmnist::read_mnist_labels(true);
   let test_images = loadmnist::read_mnist_images(false);
   let test_labels = loadmnist::read_mnist_labels(false);

   let p = model::Hyperparams {
      epochs: 1,
      batch_size: 32,
      learn_rate: 0.01,
   };

   let relu = ReLU;
   let softmax = soft_max;

   let mut model = model::Model::new(relu, softmax);

   for i in 0..10 {
      println!("--- Epoch {} ---", i);
      let now = std::time::Instant::now();
      model.test(&test_images, image_height * image_width, num_test_image, &test_labels);
      let elapsed = now.elapsed();
      println!("Test time: {:.?}" , elapsed);

      let now = std::time::Instant::now();
      model.train(&train_images, image_height * image_width, num_train_images, &train_labels, &p);
      let elapsed = now.elapsed();
      println!("Epoch training time: {:.?}", elapsed);
   }

   let now = std::time::Instant::now();
   model.test(&test_images, image_height * image_width, num_test_image, &test_labels);
   let elapsed = now.elapsed();
   println!("Test time: {:.?}" , elapsed);

}