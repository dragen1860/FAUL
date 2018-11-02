import  os
import  numpy as np
import  tensorflow as tf
import  utils


FLAGS = tf.flags.FLAGS



class FAUL:
    """
    This is our own class to support Fast Adaption on Unsupervised few-shot Learning
    """
    def __init__(self, dbs, train_dir, **kwargs):
        """

        :param dbs: dataset class containing train/test PrefetchDataset, width/height/colors
        :param train_dir: ./logs
        :param kwargs: parameters for individual model
        """
        self.train_db = dbs['train_db']
        self.test_db = dbs['test_db']
        self.imgsz = dbs['imgsz']
        self.imgc = dbs['imgc']
        self.n_way = dbs['n_way']
        self.k_spt = dbs['k_spt']
        self.k_qry = dbs['k_qry']
        # logs/mnist32/AEBaseline_depth64_latent16_scales3
        self.train_dir = os.path.join(train_dir, dbs['name'], self.experiment_name(**kwargs))
        # extra parameters for each individual model
        self.params = kwargs

        # create checkpoint directory: tf, summary director:summary, image directory: image
        for dir in (self.checkpoint_dir, self.summary_dir, self.image_dir):
            if not os.path.exists(dir):
                os.makedirs(dir)

        for k, v in kwargs.items():
            print(k, v)


        self.train_graph = None
        self.test_graph = None


    def experiment_name(self, **kwargs):
        """
        Compose a string indicating the current experiment name with related hyperparameters
        :param kwargs:
        :return:
        """
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        # AEBaseline_depth64_latent16_scales3
        return '_'.join([self.__class__.__name__] + args)


    @property
    def image_dir(self):
        return os.path.join(self.train_dir, 'images')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'ckpts')

    @property
    def summary_dir(self):
        return os.path.join(self.checkpoint_dir, 'summaries')

    @property
    def tf_sess(self):
        """
        This method return with unsupervised session, compared with self.sess,
        which is a tf.train.MonitoredTrainingSession
        :return:
        """
        return self.sess._tf_sess()

    @staticmethod
    def add_summary_var(name):
        """
        add variable name into summary, and name it 'name' in summary
        :param name:
        :return:
        """
        # this will create a new variable.
        v = tf.get_variable(name, [], trainable=False, initializer=tf.initializers.zeros())
        tf.summary.scalar(name, v)
        return v


    def eval_mode(self):
        """
        create a new eval graph and put all ops on this graph
        and restore model from ckpt
        :return:
        """
        if self.eval_graph is None:
            self.eval_graph = tf.Graph()

        with self.eval_graph.as_default():
            global_step = tf.train.get_or_create_global_step()
            # this model function will be implemented by child class!
            self.eval_ops = self.model(**self.params)
            self.eval_sess = tf.Session()
            saver = tf.train.Saver()

            ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
            saver.restore(self.eval_sess, ckpt)
            tf.logging.info('Restore  %s from ckpt, eval mode at global_step %d',
                                self.__class__.__name__, self.eval_sess.run(global_step))


    def train(self):
        """

        :param report_kimg: report at every report_kimg
        :return:
        """
        batchsz = FLAGS.batchsz
        report_kimg = 1 << 6


        if self.train_graph is None:
            self.train_graph = tf.Graph()

        with self.train_graph.as_default():

            # data_in = self.train_data.make_one_shot_iterator().get_next()
            global_step = tf.train.get_or_create_global_step()


            some_float = tf.placeholder(tf.float32, [], 'some_float')
            self.latent_accuracy = self.add_summary_var('latent_accuracy')
            update_summary_var = lambda x: tf.assign(x, some_float)
            latent_accuracy_op = update_summary_var(self.latent_accuracy)

            summary_hook = tf.train.SummarySaverHook(
                                save_steps=(report_kimg << 9) // batchsz, # save every steps
                                output_dir=self.summary_dir,
                                summary_op=tf.summary.merge_all())
            stop_hook = tf.train.StopAtStepHook(last_step=1 + (FLAGS.total_kimg << 10) // batchsz)
            report_hook = utils.HookReport(report_kimg << 7, batchsz)

            init_my_summary_op = lambda op, value: self.tf_sess.run(op, feed_dict={some_float: value})



            # main op
            ops = self.model(**self.params)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=self.checkpoint_dir, # automatically restore from ckpt
                    hooks=[stop_hook],
                    chief_only_hooks=[report_hook, summary_hook], # the hooks are only valid for chief session
                    save_checkpoint_secs=5000, # every 6 minutes save ckpt
                    save_summaries_steps=0, # do NOT save summary into checkpoint_dir
                    config=config) as sess:

                self.sess = sess
                self.cur_nimg = batchsz * self.tf_sess.run(global_step)

                while not sess.should_stop():
                    # run data_in ops first and then run ops.train_op

                    spt_x, spt_y, qry_x, qry_y = self.train_db.get_batch(batchsz, use_episode=True)

                    sess.run(ops['meta_op'], feed_dict={
                        ops['train_spt_x']: spt_x,
                        ops['train_spt_y']: spt_y,
                        ops['train_qry_x']: qry_x,
                        ops['train_qry_y']: qry_y
                    })

                    # MonitoredTrainingSession has an underlying session object: tf_session
                    # current computed number of image
                    self.cur_nimg = batchsz * self.tf_sess.run(global_step)


                    # Time to evaluate classification accuracy
                    if self.cur_nimg % (report_kimg << 6) == 0:
                        # return with float accuracy
                        accuracy = self.eval_latent_accuracy(ops)
                        # print('eval accuracy:', accuracy, self.cur_nimg)
                        # self.tf_sess.run(latent_accuracy_op, feed_dict={some_float: accuracy})
                        init_my_summary_op(latent_accuracy_op, accuracy)


    def eval_latent_accuracy(self, ops):
        """
        Eval MLP classification accuracy based on latent representation
        :param ops:
        :param batches:
        :return:
        """

        batchsz = FLAGS.batchsz
        update_num = FLAGS.update_num
        correct = np.zeros(update_num+1)
        totoal_counter = 0

        while True:
            spt_x, spt_y, qry_x, qry_y = self.test_db.get_batch(batchsz, use_episode=True)
            pretrain_op_ = self.tf_sess.run(ops['pretrain_op'],
                                                    feed_dict={
                                                        ops['test_spt_x']: spt_x,
                                                        ops['test_spt_y']: spt_y,
                                                        ops['test_qry_x']: qry_x,
                                                        ops['test_qry_y']: qry_y
                                                    })
            qry_preds = self.tf_sess.run(ops['classify_preds'],
                                                    feed_dict={
                                                        ops['test_spt_x']: spt_x,
                                                        ops['test_spt_y']: spt_y,
                                                        ops['test_qry_x']: qry_x,
                                                        ops['test_qry_y']: qry_y
                                                    })
            for i, qry_pred in enumerate(qry_preds):
                correct[i] += (qry_pred == qry_y.argmax(1))
            totoal_counter += batchsz

            if totoal_counter > 1000:
                break

        accuracy = correct / totoal_counter
        tf.logging.info('Eval acc:', accuracy)
        return accuracy


    def make_sample_grid_and_save(self, ops, batch_size=16, random=4, interpolation=16, height=16, save_to_disk=True):
        """

        :param ops: AEops class, including train_op
        :param batch_size:
        :param random: number of reconstructed images = random * height
        :param interpolation: number of interpolation, namely the row number of compositive image
        :param height: number of hight
        :param save_to_disk:
        :return: recon, inter, slerp, samples
        """
        # Gather images
        pool_size = random * height + 2 * height # 96
        current_size = 0

        with tf.Graph().as_default():
            data_in = self.test_data.make_one_shot_iterator().get_next()
            with tf.Session() as sess_new:
                images = []
                while current_size < pool_size:
                    images.append(sess_new.run(data_in)['x'])
                    current_size += images[-1].shape[0]
                images = np.concatenate(images, axis=0)[:pool_size] # [96, 32, 32, 1]

        def batched_op(op, op_input, array):
            return np.concatenate(
                [
                    self.tf_sess.run(op, feed_dict={
                        op_input: array[x:x + batch_size]})
                    for x in range(0, array.shape[0], batch_size)
                ],
                axis=0)

        # 1. Random reconstructions
        if random: # not zero
            random_x = images[:random * height] # [64, 32, 32, 1]
            random_y = batched_op(ops.ae, ops.x, random_x)
            randoms = np.concatenate([random_x, random_y], axis=2) # ae output: [64, 32, 32, 1] => [64, 32, 64, 1]
            image_random = utils.images_to_grid( # [16, 4, 32, 64, 1] => [512, 256, 1]
                randoms.reshape((height, random) + randoms.shape[1:]))
        else:
            image_random = None

        # 2. Interpolations
        interpolation_x = images[-2 * height:] # [32, 32, 32, 1]
        latent_x = batched_op(ops.encode, ops.x, interpolation_x) # [32, 4, 4, 16]
        latents = []
        for x in range(interpolation):
            latents.append((latent_x[:height] * (interpolation - x - 1) +
                            latent_x[height:] * x) / float(interpolation - 1))
        latents = np.concatenate(latents, axis=0) # [256, 4, 4, 16]
        interpolation_y = batched_op(ops.decode, ops.h, latents) # [256, 32, 32, 1]
        interpolation_y = interpolation_y.reshape( # [16, 16, 32, 32, 1]
            (interpolation, height) + interpolation_y.shape[1:])
        interpolation_y = interpolation_y.transpose(1, 0, 2, 3, 4)
        image_interpolation = utils.images_to_grid(interpolation_y) # [512, 512, 1]

        # 3. Interpolation by slerp
        latents_slerp = []
        dots = np.sum(latent_x[:height] * latent_x[height:],
                      tuple(range(1, len(latent_x.shape))),
                      keepdims=True) # [16, 1, 1, 1]
        norms = np.sum(latent_x * latent_x,
                       tuple(range(1, len(latent_x.shape))),
                       keepdims=True) # [32, 1, 1, 1]
        cosine_dist = dots / np.sqrt(norms[:height] * norms[height:]) # [16, 1, 1, 1]
        omega = np.arccos(cosine_dist)
        for x in range(interpolation):
            t = x / float(interpolation - 1)
            latents_slerp.append(
                np.sin((1 - t) * omega) / np.sin(omega) * latent_x[:height] +
                np.sin(t * omega) / np.sin(omega) * latent_x[height:])
        latents_slerp = np.concatenate(latents_slerp, axis=0) # 16 of[16, 4, 4, 16] => [256, 4, 4, 16]
        interpolation_y_slerp = batched_op(ops.decode, ops.h, latents_slerp) # [256, 32, 32, 1]
        interpolation_y_slerp = interpolation_y_slerp.reshape(
            (interpolation, height) + interpolation_y_slerp.shape[1:]) # [16, 16, 32, 32, 1]
        interpolation_y_slerp = interpolation_y_slerp.transpose(1, 0, 2, 3, 4)
        image_interpolation_slerp = utils.images_to_grid(interpolation_y_slerp) # [512, 512, 1]

        # 4. get decoder by random normal dist of hidden h
        random_latents = np.random.standard_normal(latents.shape) # [256, 4, 4, 16]
        samples_y = batched_op(ops.decode, ops.h, random_latents)
        samples_y = samples_y.reshape(
            (interpolation, height) + samples_y.shape[1:])
        samples_y = samples_y.transpose(1, 0, 2, 3, 4)
        image_samples = utils.images_to_grid(samples_y) # [512, 512, 1]

        if random: # [512, 256+512+512+512, 1]
            image = np.concatenate(
                [image_random, image_interpolation, image_interpolation_slerp,
                 image_samples], axis=1)
        else:
            image = np.concatenate(
                [image_interpolation, image_interpolation_slerp,
                 image_samples], axis=1)
        if save_to_disk:
            utils.save_images(utils.to_png(image), self.image_dir, self.cur_nimg)

        return image_random, image_interpolation, image_interpolation_slerp, image_samples





    def model(self, **kwargs):
        raise NotImplementedError




