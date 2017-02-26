import os
import re
import shutil
import time

import gym
import numpy as np
import tensorflow as tf

import tools

tf.app.flags.DEFINE_string('base_dir', '', 'Base directory to save summaries and checkpoints')

tf.app.flags.DEFINE_string('env', 'CartPole-v1', 'Name of environment to run')
tf.app.flags.DEFINE_float('lr', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_float('clip_grad', 10., 'Gradients norm to clip gradients to')
tf.app.flags.DEFINE_integer('steps', 10 * 10 ** 6, 'Number of steps to run learning for')
tf.app.flags.DEFINE_integer('steps_per_action', 5, 'How many NN updates per one env action')
tf.app.flags.DEFINE_boolean('restart', True,
                            'If true, starts over, otherwise, starts from the last checkpoint')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')

# Parameters of replay buffer
# Specification of buffer. There are two options, prioritized and simple
# simple has format 's(\d+)', where the 2 ^ (Number) is the length of buffer
# prioritized has format 'p(\d+).(\d+).(\d+)'. First number specifies size of the buffer
# (see above). Second and third numbers define alpha and beta,
# where leading '0.' are dropped
tf.app.flags.DEFINE_string('buffer', 's17', 'Experience buffer spec.')
tf.app.flags.DEFINE_integer('init_buffer_size', 10 ** 5,
                            'Minimum number of replays to start learning')

# Parameters of experience generation
# tf.app.flags.DEFINE_string('experience', '20.99', 'Experience generation .')

# Exploration specification
# e-3-01-1M - epsilon-greedy, starts at .3 and decreases until .01 for 1M steps.
# t-10.1-0.01-1M - softmax based, with temperature starting at 10.1, decreasing
# to 0.01 for 1M steps
tf.app.flags.DEFINE_string('exploration', '', 'Experience buffer spec.')

tf.app.flags.DEFINE_integer('summary_every_steps', 250, 'The frequence of summarizing stuff')

tf.app.flags.DEFINE_boolean('image_summaries', True,
                            'If True, periodically renders environments to TF summaries')

FLAGS = tf.app.flags.FLAGS

def Select(value, index):
    # Value - float tensor of (batch, actions) size
    # index - int32 tensor of (batch) size
    # returns float tensor of batch size where in every batch the element from index is selected
    batch_size = tf.shape(value)[0]
    batch = tf.range(0, batch_size)
    ind = tf.concat([tf.expand_dims(batch, 1),
                     tf.expand_dims(index, 1)], 1)
    return tf.gather_nd(value, ind)


def CartPoleQNetwork(state, num_actions, unused_is_training):
    hidden = tf.contrib.layers.fully_connected(
        state, 64,
        activation_fn=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.005),
        scope='hidden1')
    tf.contrib.layers.summarize_tensor(hidden)
    hidden = tf.contrib.layers.fully_connected(
        hidden, 64,
        activation_fn=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.005),
        scope='hidden2')
    tf.contrib.layers.summarize_tensor(hidden)
    hidden = tf.contrib.layers.fully_connected(
        hidden, 64,
        activation_fn=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.005),
        scope='hidden3')
    tf.contrib.layers.summarize_tensor(hidden)

    value = tf.contrib.layers.linear(
        hidden, 1,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        biases_initializer=tf.constant_initializer(0.),
        scope='value')
    tf.contrib.layers.summarize_tensor(value)
    adv = tf.contrib.layers.linear(hidden, num_actions,
                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                   scope='advantage')
    adv = tf.subtract(adv, tf.reduce_mean(adv, reduction_indices=1, keep_dims=True), 'advantage')
    tf.contrib.layers.summarize_tensor(adv)

    output = tf.add(value, adv, 'output')
    return output


ROLLOUT_LEN = 1 # 20
GAMMA = 0.99
UPDATE_STEPS = 10000


def ReplayBufferFactory(spec):
    regex = re.compile(r'(s(?P<s_num>\d+))|(p(?P<p_num>\d+)\.(?P<alpha>\d+)\.(?P<beta>\d+))')
    match = regex.match(spec)
    if match is None:
        raise ValueError('Invalid replay buffer specification %s' % spec)
    if match.group('s_num'):
        return tools.ExperienceBuffer(1 << int(match.group('s_num')))
    else:
        alpha = float('0.%s' % match.group('alpha'))
        beta = float('0.%s' % match.group('beta'))
        size = int(match.group('p_num'))
        return tools.WeightedExperienceBuffer(alpha, beta, max_weight=100.,
                                              buffer_size=1 << size)

def PolicyFactory(spec, qvalues, global_step):
    """Takes exploration spec and qvalues tensor, returns policy tensor"""
    regex = re.compile(r'(e-(?P<eps_start>\d+)-(?P<eps_end>\d+)-(?P<eps_steps>[\d\.]+)M)|' +
                       r'(t-(?P<tmp_start>[\d\.]+)-(?P<tmp_end>[\d\.]+)-(?P<tmp_steps>[\d\.]+)M)')
    m = regex.match(spec)
    if m is None:
        raise ValueError('Invalid exploration spec %s' % spec)

    float_step = tf.cast(global_step, tf.float32)
    if m.group('eps_start'):
        # Epsilon greedy
        start = float('0.%s' % m.group('eps_start'))
        end = float('0.%s' % m.group('eps_end'))
        steps = float(m.group('eps_steps')) * 10 ** 6

        epsilon = tf.maximum(end, start + (end - start) * float_step / steps)
        tf.summary.scalar("Scalars/Epsilon", epsilon)

        coin = tf.random_uniform(shape=[], maxval=1)
        policy = tf.cond(coin < epsilon,
                         lambda: tf.reshape(tf.multinomial(tf.zeros_like(qvalues), 1), []),
                         lambda: tf.reshape(tf.argmax(qvalues, axis=1), []))

        n_actions = tf.cast(tf.shape(qvalues)[1], tf.float32)
        min_prob = epsilon / n_actions
        max_prob = (1 - epsilon) + min_prob
        tf.summary.scalar("Scalars/Prob/Max", max_prob)
        tf.summary.scalar("Scalars/Prob/Min", min_prob)

        entropy = -((n_actions - 1) * tf.log(min_prob) * min_prob + tf.log(max_prob) * max_prob)
    else:
        # Softmax
        start = float(m.group('tmp_start'))
        end = float(m.group('tmp_end'))
        steps = float(m.group('tmp_steps')) * 10 ** 6

        temperature = tf.maximum(end, start + (end - start) * float_step / steps)
        tf.summary.scalar("Scalars/Temperature", temperature)

        logits = qvalues / temperature
        policy = tf.reshape(tf.multinomial(logits, 1), [])

        tf.summary.scalar("Scalars/Prob/Max", tf.reduce_max(tf.nn.softmax(logits)))
        tf.summary.scalar("Scalars/Prob/Min", tf.reduce_min(tf.nn.softmax(logits)))

        entropy = tf.reduce_sum(-tf.nn.softmax(logits) * tf.nn.log_softmax(logits))

    tf.summary.scalar("Scalars/Entropy", entropy)
    return policy


def GenerateExperience(env, policy, step_callback, stats_callback):
    episode_rew = 0.
    episode_len = 0.
    old_s = env.reset()
    while True:
        ss, aa, rr, ss1, gg = [], [], [], [], []
        done = False
        while not done and len(ss) < ROLLOUT_LEN:
            a = policy(old_s)

            s, r, done, _ = env.step(a)
            ss.append(old_s)
            aa.append(a)
            rr.append(r)
            ss1.append(s)
            gg.append(GAMMA if not done else 0.)

            episode_rew += r
            episode_len += 1
            old_s = s

        rew = 0.
        g = 1.
        for i in reversed(range(len(ss))):
            rew = rr[i] + gg[i] * rew
            g *= gg[i]
            ss1[i] = old_s
            rr[i] = r
            gg[i] = g

        if done:
            old_s = env.reset()
            stats_callback(episode_rew, episode_len)
            episode_rew, episode_len = 0., 0.

        should_continue = step_callback(np.array(ss), np.array(aa), np.array(rr),
                                        np.array(ss1), np.array(gg), 100)
        if not should_continue:
            return


def InitSession(sess, folder):
    """If folder has checkpoint, reinitializes session with it"""
    last_step = -1
    if not FLAGS.restart:
        for fname in os.listdir(folder):
            m = re.match(r'model.ckpt-(\d+).meta', fname)
            if m:
                step = int(m.group(1))
                if step > last_step:
                    last_step = step

    saver = tf.train.Saver()
    if last_step > 0:
        saver.restore(sess, os.path.join(folder, 'model.ckpt-%d' % last_step))
    else:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        sess.run(tf.global_variables_initializer())

    return saver

def main(argv):
    folder = os.path.join(FLAGS.base_dir, FLAGS.env, 'lr-%.1E' % FLAGS.lr,
                          FLAGS.buffer,
                          'bs-%d' % FLAGS.batch_size,
                          FLAGS.exploration)

    env = gym.make(FLAGS.env)
    buf = ReplayBufferFactory(FLAGS.buffer)
    def FillBuffer(*args):
        buf.add(*args)
        return buf.inserted < FLAGS.init_buffer_size

    while buf.inserted < FLAGS.init_buffer_size:
        GenerateExperience(env, lambda _: env.action_space.sample(),
                           FillBuffer, lambda *args: None)

    state2q = CartPoleQNetwork

    state = tf.placeholder(tf.float32, shape=[None, buf.state_size], name='state')
    action = tf.placeholder(tf.int32, shape=[None], name='action')
    reward = tf.placeholder(tf.float32, shape=[None], name='reward')
    state1 = tf.placeholder(tf.float32, shape=[None, buf.state_size], name='state1')
    gamma = tf.placeholder(tf.float32, shape=[None], name='gamma')
    is_weights = tf.placeholder(tf.float32, shape=[None], name='is_weights')
    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')

    with tf.variable_scope('model', reuse=False):
        qvalues = state2q(state, env.action_space.n, is_training)
    with tf.variable_scope('model', reuse=True):
        qvalues1 = state2q(state1, env.action_space.n, is_training)
    with tf.variable_scope('target', reuse=False):
        qvalues_target = state2q(state1, env.action_space.n, is_training)

    vars_pred = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'model')
    vars_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target')

    # Copies variables to target network
    copy_op = tf.group(*[tf.assign(y, x) for x, y in zip(vars_pred, vars_target)])

    act_s1 = tf.cast(tf.argmax(qvalues1, dimension=1), tf.int32)
    q_s1 = Select(qvalues_target, act_s1)
    target_q = tf.stop_gradient(reward + gamma * q_s1)
    q = Select(qvalues, action)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    policy = PolicyFactory(FLAGS.exploration, qvalues, global_step)

    delta = target_q - q
    td_err_weight = tf.abs(delta)
    loss = tf.reduce_mean(tools.HuberLoss(delta, 5) * is_weights)
    tf.summary.histogram('Monitor/TD_Error', delta)
    tf.summary.histogram('Monitor/Q', q)
    tf.summary.histogram('Monitor/Weights', is_weights)
    tf.summary.scalar("Scalars/Q", tf.reduce_mean(q))
    tf.summary.scalar('Scalars/Weights', tf.reduce_mean(is_weights))
    tf.summary.scalar("Scalars/Total_Loss", loss)

    if FLAGS.image_summaries:
        test_image = env.render(mode='rgb_array')
        render_image = tf.placeholder(tf.uint8, shape=test_image.shape, name='render')
        tf.summary.image('Render', tf.expand_dims(render_image, 0), max_outputs=3)

    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')
    grads = optimizer.compute_gradients(loss, variables)
    grads = tools.ClipGradient(grads, FLAGS.clip_grad)

    for grad, v in grads:
        if grad is not None:
            tf.summary.histogram('{}/grad'.format(v.name), grad)
        # tf.summary.histogram(v.name, v)

    train_op = tf.group(optimizer.apply_gradients(grads, global_step),
                        *tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    tf.contrib.layers.summarize_activations()
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        saver = InitSession(sess, folder)
        writer = tf.summary.FileWriter(folder)
        writer.add_graph(tf.get_default_graph())

        steps = {'action': 0, 'time': time.time()}

        def Policy(obs):
            return sess.run(policy, {state: np.reshape(obs, (1, -1)), is_training: False})

        def Stat(reward, length):
            writer.add_summary(tf.Summary(
                value=[tf.Summary.Value(tag='Env/Reward', simple_value=reward),
                       tf.Summary.Value(tag='Env/Length', simple_value=length)]),
                       sess.run(global_step))

        def Step(*buf_args):
            buf.add(*buf_args)
            steps['action'] += 1
            for _ in xrange(FLAGS.steps_per_action):
                idx, ss, aa, rr, ss1, gg, ww = buf.sample(FLAGS.batch_size)
                if ss is None:
                    return True

                feed_dict = {state: ss, action: aa, reward: rr, state1: ss1,
                             gamma: gg, is_weights: ww, is_training: True}

                cur_step = sess.run(global_step)
                if cur_step > FLAGS.steps:
                    return False  # Time to stop

                if cur_step % FLAGS.summary_every_steps != 0:
                    weights, _ = sess.run([td_err_weight, train_op], feed_dict)
                else:
                    if FLAGS.image_summaries:
                        feed_dict[render_image] = env.render(mode='rgb_array')
                    weights, _, smr = sess.run(
                        [td_err_weight, train_op, summary_op], feed_dict)
                    writer.add_summary(smr, cur_step)

                for ii, td_w in zip(idx, weights):
                    buf.tree_update(ii, td_w)

                if cur_step % UPDATE_STEPS == 0 and cur_step > 0:
                    print 'Updated target network (%d)' % cur_step
                    sess.run(copy_op)
                    saver.save(sess, os.path.join(folder, 'model.ckpt'), global_step=global_step)
                    writer.add_summary(tf.Summary(
                        value=[tf.Summary.Value(
                            tag='Steps per sec',
                            simple_value=UPDATE_STEPS / (time.time() - steps['time']))]),
                        cur_step)
                    steps['time'] = time.time()
            return True

        GenerateExperience(env, Policy, Step, Stat)


if __name__ == "__main__":
    tf.app.run(main)
