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

from cleverhans.loss import CrossEntropy
from cleverhans.model import Model
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import MadryEtAl
from cleverhans.utils import AccuracyReport, set_log_level

import CNN_module as CNN

#Temporarily disable deprecation warnings (using tf 1.14)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

class native_cleverhans_model(Model):
  def __init__(self, 
                scope, 
                nb_classes, 
                dropout_rate_placeholder,
                dynamic_var,
                **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.dropout_rate_placeholder = dropout_rate_placeholder
    self.dynamic_var = dynamic_var

    print("\nBuilding a " + self.scope + " model for adversarial training.")

    self.model_prediction_function = getattr(CNN, self.scope + '_predictions')
    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, [128, 28, 28, 1]))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

        initializer = tf.contrib.layers.variance_scaling_initializer()

        self.weights = {
        'conv_W1' : tf.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer),
        'conv_W2' : tf.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer),
        'dense_W1' : tf.get_variable('DW1', shape=(400, 120), initializer=initializer),
        'dense_W2' : tf.get_variable('DW2', shape=(120, 84), initializer=initializer),
        'output_W' : tf.get_variable('OW', shape=(84, 10), initializer=initializer)
        }
        if (self.scope == 'BindingCNN'):
          self.weights['course_bindingW1'] = tf.get_variable('courseW1', shape=(1600, 120), initializer=initializer)
          self.weights['finegrained_bindingW1'] = tf.get_variable('fineW1', shape=(1176, 120), initializer=initializer)

        self.biases = {
        'conv_b1' : tf.get_variable('Cb1', shape=(6), initializer=initializer),
        'conv_b2' : tf.get_variable('Cb2', shape=(16), initializer=initializer),
        'dense_b1' : tf.get_variable('Db1', shape=(120), initializer=initializer),
        'dense_b2' : tf.get_variable('Db2', shape=(84), initializer=initializer),
        'output_b' : tf.get_variable('Ob', shape=(10), initializer=initializer)
        }

        logits, _, _, _ = self.model_prediction_function(x, self.dropout_rate_placeholder, self.weights, self.biases, self.dynamic_var)

        return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}


def adver_training(model_params, iter_num, training_data, 
          training_labels, evaluation_data, evaluation_labels):
  """
  Based on the MNIST cleverhans tutorial (https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/mnist_tutorial_tf.py)
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()
  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  x_placeholder = tf.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
  y_placeholder = tf.placeholder(training_labels.dtype, [None, 10], name='y-input')
  dropout_rate_placeholder = tf.placeholder(tf.float32)
  nb_classes = 10

  sess = tf.Session()

  def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x_placeholder, y_placeholder, preds, x_set, y_set, args=eval_params, feed={dropout_rate_placeholder: 0.0})
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))
  
  ch_model = native_cleverhans_model(scope=model_params['architecture'], 
                                  nb_classes=nb_classes, 
                                  dropout_rate_placeholder=dropout_rate_placeholder,
                                  dynamic_var=model_params['dynamic_var'])

  Madry_attack = MadryEtAl(ch_model, sess=sess)
  
  def attack(x):
    return Madry_attack.generate(x)

  saver = tf.train.Saver(ch_model.get_params())

  loss = CrossEntropy(ch_model, smoothing=model_params['label_smoothing'], attack=attack)
  preds = ch_model.get_logits(x_placeholder)
  adv_x_Madry = attack(x_placeholder)
  preds_adv_Madry = ch_model.get_logits(adv_x_Madry)

  def evaluate():
    # Accuracy of adversarially trained model on legitimate test inputs
    do_eval(preds, evaluation_data, evaluation_labels, 'adv_train_clean_eval', False)
    # Accuracy of the adversarially trained model on adversarial examples
    do_eval(preds_adv_Madry, evaluation_data, evaluation_labels, 'adv_train_adv_eval', True)

  # Perform and evaluate adversarial training
  train_params = {
      'nb_epochs': model_params['training_epochs'],
      'batch_size': model_params['batch_size'],
      'learning_rate': model_params['learning_rate']
  }
  eval_params = {'batch_size': model_params['batch_size']}

  train(sess, loss, training_data, training_labels, evaluate=evaluate,
        feed={dropout_rate_placeholder: model_params['dropout_rate']}, args=train_params, var_list=ch_model.get_params())

  network_name_str = str(iter_num) + model_params['architecture'] + '_adver_trained_' + str(model_params['adver_trained'])
  save_path = saver.save(sess, "network_weights_data/" + network_name_str + ".ckpt")

  final_test_acc = model_eval(sess, x_placeholder, y_placeholder, preds, evaluation_data, evaluation_labels, args=eval_params, feed={dropout_rate_placeholder: 0.0})

  return final_test_acc, network_name_str

