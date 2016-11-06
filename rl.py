import threading
import time

import numpy as np
import numpy.random
import tensorflow as tf


def discounted_rewards(rewards, gamma, bootstrap):
    rank = rewards.get_shape().ndims or tf.rank(rewards)
    reverse_params = tf.concat(0, [
            tf.constant([True]),
            tf.fill(tf.expand_dims(rank - 1, 0), False)
        ])
    reverse_rew = tf.reverse(rewards, reverse_params)
    
    summed = tf.scan(lambda a, x: x + gamma * a, reverse_rew, initializer=bootstrap,
                     parallel_iterations=1, back_prop=False)
    return tf.reverse(summed, reverse_params)


def Last(bb):
    """Returs last element of given 1d tensor."""
    size = tf.shape(bb)[0]
    return tf.reshape(tf.gather(bb, size - 1), [])


DEFAULT_OPTIONS = {
    'clip_grad': 5.,
    'gamma': 0.99, # Discount factor
    'learning_rate': 0.0001,
    'rollout': 20, # Maximum length of each experience
    'threads': 1,  # Number of threads to generate experiences
}


class ActorCritic(object):
    def __init__(self, env, build_networks, options=DEFAULT_OPTIONS):
        self._env = env
        self._env_example = env()
        self._options = options
        state_dim = self._env_example.observation_space.shape
        self._graph = tf.Graph()
        self._threads = []
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._state = tf.placeholder(tf.float32, shape=[None] + list(state_dim) , name='states')
            self._action = tf.placeholder(tf.int64, shape=[None], name='actions')
            self._reward = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')
            self._done = tf.placeholder(tf.float32, shape=[1], name='done')

            self._policy_logits, self._baseline = build_networks(self._env_example, self._state)

            self._discount = discounted_rewards(self._reward, options['gamma'],
                                                Last(self._baseline) * (1. - self._done))

            self._tf_policy = tf.reshape(tf.multinomial(self._policy_logits, 1), [])

            with tf.device('/cpu:0'):
                optimizer = tf.train.AdamOptimizer(options['learning_rate'])

            advantage = tf.reshape(self._discount, [-1, 1]) - self._baseline

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self._policy_logits, tf.reshape(self._action, [-1]))
            policy_loss = tf.reduce_mean(tf.mul(cross_entropy, tf.stop_gradient(advantage)))
            policy_entropy = tf.reduce_mean(-tf.nn.softmax(self._policy_logits) *
                                            tf.nn.log_softmax(self._policy_logits))
            value_loss = 0.5 * tf.reduce_mean(tf.square(advantage))

            loss = policy_loss + 0.25 * value_loss - 0.01 * policy_entropy

            grads = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.VARIABLES))
            if 'clip_grad' in options:
                grads = [(tf.clip_by_norm(g, options['clip_grad']), v)
                         for g, v in grads]

            for grad, var in grads:
                tf.histogram_summary(var.name, var)
                if grad is not None:
                    tf.histogram_summary('{}/grad'.format(var.name), grad)

            self._global_step = tf.Variable(0, name='global_step', trainable=False)
            self._train_op = optimizer.apply_gradients(grads, self._global_step)

            tf.histogram_summary("Predicted baseline", self._baseline)
            tf.scalar_summary("Loss/Actor", policy_loss)
            tf.scalar_summary("Loss/Critic", value_loss)
            tf.scalar_summary("Loss/Entropy", policy_entropy)
            tf.scalar_summary("Loss/Total", loss)

            self._summary_op = tf.merge_all_summaries()

            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self._last_start = None

    def Init(self, logs_dir):
        with self._graph.as_default():
            self.sess.run(tf.initialize_all_variables())
            self._writer = tf.train.SummaryWriter(logs_dir)
            self._coord = tf.train.Coordinator()

    def Close(self):
        self._coord.request_stop()
        self._coord.join(self._threads)
        self.sess.close()

    def Policy(self, observation):
        return self.sess.run(
            self._tf_policy,
            {self._state: observation.reshape(
                [1] + list(self._env_example.observation_space.shape))})

    def Learn(self, num_steps):
        if any([t.is_alive() for t in self._threads]):
            print 'At least one thread is already running!'
            return

        self._threads = [threading.Thread(target=self.LearnThread, args=(num_steps,))
                         for _ in xrange(self._options['threads'])]
        for t in self._threads:
            t.start()

    def LearnThread(self, num_steps):
        env = self._env()
        obs = env.reset()

        observations, actions, rewards = [], [], []
        done = False
        episode_reward, episode_len = 0., 0.

        while not self._coord.should_stop():
            observations.append(obs)
            act = self.Policy(obs)
            actions.append(act)

            obs, reward, done, _ = env.step(act)
            episode_reward += reward
            episode_len += 1.
            rewards.append(reward)

            if done or len(observations) >= self._options.get('rollout', 20):
                step = self.Update(observations, actions, rewards, done)
                if done:
                    obs = env.reset()
                    done = False
                    self._writer.add_summary(tf.Summary(value=[
                        tf.Summary.Value(tag='Env/Rewards', simple_value=episode_reward),
                        tf.Summary.Value(tag='Env/Length', simple_value=episode_len)]), step)
                    episode_reward, episode_len = 0., 0.

                observations, actions, rewards = [], [], []

                if step >= num_steps:
                    print 'Many steps!'
                    self._coord.request_stop()

    def Update(self, observations, actions, rewards, done):
        feed_dict = {self._state: observations,
                     self._action: actions,
                     self._reward: np.reshape(rewards, (-1, 1)),
                     self._done: [1.0 if done else 0.0]}

        step, _ = self.sess.run([self._global_step, self._train_op], feed_dict)

        if step % 50 == 0:
            self._writer.add_summary(self.sess.run(self._summary_op, feed_dict), step)

        if step % 1000 == 0:
            if self._last_start:
                self._writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(
                        tag='Steps per sec',
                        simple_value=1000./(time.time() - self._last_start))]), step)
            self._last_start = time.time()

        return step
