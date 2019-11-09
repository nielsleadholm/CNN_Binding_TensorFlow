#!/usr/bin/env python3

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import os

from cleverhans.compat import flags
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import DeepFool
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

from tfCore_adversarial_attacks import native_cleverhans_model
from tfCore_adversarial_attacks import custom_cleverhans_loss
import CNN_module as CNN



#Temporarily disable deprecation warnings (using tf 1.14)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 



FLAGS = flags.FLAGS

NB_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = False
BACKPROP_THROUGH_ATTACK = True
NB_FILTERS = 64


def adver_training(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE,
                   clean_train=CLEAN_TRAIN,
                   testing=False,
                   backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                   nb_filters=NB_FILTERS, num_threads=None,
                   label_smoothing=0.1):
  """
  MNIST cleverhans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
  :param testing: if true, complete an AccuracyReport for unit tests
                  to verify that performance is adequate
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()
  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get MNIST data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  dropout_rate_placeholder = tf.placeholder(tf.float32)

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  eval_params = {'batch_size': batch_size}
  fgsm_params = {
      'eps': 0.3,
      'clip_min': 0.,
      'clip_max': 1.
  }
  rng = np.random.RandomState([2017, 8, 30])

  def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, feed={dropout_rate_placeholder : 0.0}, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  model_params = {
    'architecture':'BindingCNN',
    'meta_architecture':'CNN',
    'dataset':'mnist',
    'dropout_rate':0.25,
    'dynamic_var':0
    }
  
  # if clean_train:
  #   model = native_cleverhans_model(scope='model1', 
		# 			                nb_classes=10, 
		# 			                model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
  #                                   dropout_rate_placeholder=dropout_rate_placeholder,
		# 			                meta_architecture=model_params['meta_architecture'])

  #   preds = model.get_logits(x)
  #   loss = custom_cleverhans_loss(model, smoothing=label_smoothing)

  #   def evaluate():
  #     do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

  #   train(sess, loss, x_train, y_train, evaluate=evaluate,
  #         feed={dropout_rate_placeholder : model_params['dropout_rate']}, args=train_params, rng=rng, var_list=model.get_params())

  #   # Calculate training error
  #   if testing:
  #     do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

  #   # Initialize the Fast Gradient Sign Method (FGSM) attack object and
  #   # graph


  #   #fgsm = FastGradientMethod(model, sess=sess)
  #   #adv_x = fgsm.generate(x, **fgsm_params)
  #   Madry_attack = MadryEtAl(model, sess=sess)
  #   adv_x = Madry_attack.generate(x)
  #   # DeepFool_attack = DeepFool(model, sess=sess)
  #   # adv_x = DeepFool_attack.generate(x)


  #   preds_adv = model.get_logits(adv_x)

  #   # Evaluate the accuracy of the MNIST model on adversarial examples
  #   do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)

  #   # Calculate training error
  #   if testing:
  #     do_eval(preds_adv, x_train, y_train, 'train_clean_train_adv_eval')

  #   print('Repeating the process, using adversarial training')

  # Create a new model and train it to be robust to FastGradientMethod
  model2 = native_cleverhans_model(scope=model_params['architecture'], 
                                  nb_classes=10, 
                                  model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
                                  dropout_rate_placeholder=dropout_rate_placeholder,
                                  dynamic_var=model_params['dynamic_var'])
  fgsm2 = FastGradientMethod(model2, sess=sess)
  Madry_attack2 = MadryEtAl(model2, sess=sess)

  saver = tf.train.Saver(model2.get_params())


  def attack(x):
    
    #return fgsm2.generate(x, **fgsm_params)
    return Madry_attack2.generate(x)

  loss2 = custom_cleverhans_loss(model2, smoothing=label_smoothing, attack=attack)
  preds2 = model2.get_logits(x)
  adv_x2_Madry = attack(x)
  adv_x2_fgsm = fgsm2.generate(x, **fgsm_params)

  if not backprop_through_attack:
    # For the fgsm attack used in this tutorial, the attack has zero
    # gradient so enabling this flag does not change the gradient.
    # For some other attacks, enabling this flag increases the cost of
    # training, but gives the defender the ability to anticipate how
    # the atacker will change their strategy in response to updates to
    # the defender's parameters.
    print("Stopping gradients for attack.")
    adv_x2_Madry = tf.stop_gradient(adv_x2_Madry)
  preds2_adv_Madry = model2.get_logits(adv_x2_Madry)
  preds2_adv_fgsm = model2.get_logits(adv_x2_fgsm)

  def evaluate2():
    # Accuracy of adversarially trained model on legitimate test inputs
    do_eval(preds2, x_test, y_test, 'adv_train_clean_eval', False)
    # Accuracy of the adversarially trained model on adversarial examples
    do_eval(preds2_adv_Madry, x_test, y_test, 'adv_train_adv_eval', True)
    do_eval(preds2_adv_fgsm, x_test, y_test, 'adv_train_adv_eval', True)

  # Perform and evaluate adversarial training
  train(sess, loss2, x_train, y_train, evaluate=evaluate2,
        feed={dropout_rate_placeholder : model_params['dropout_rate']}, args=train_params, rng=rng, var_list=model2.get_params())

  save_path = saver.save(sess, "network_weights_data/adver_trained.ckpt")

  # Calculate training errors
  if testing:
    do_eval(preds2, x_train, y_train, 'train_adv_train_clean_eval')
    do_eval(preds2_adv_Madry, x_train, y_train, 'train_adv_train_adv_eval')
    do_eval(preds2_adv_fgsm, x_test, y_test, 'train_adv_train_adv_eval')


  return report


def main(argv=None):
  """
  Run the tutorial using command line flags.
  """
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 clean_train=FLAGS.clean_train,
                 backprop_through_attack=FLAGS.backprop_through_attack,
                 nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))

  tf.app.run()