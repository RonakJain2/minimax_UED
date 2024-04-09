train_steps = 0
# for i in range(N_UPDATES):
# 	start = time.time()
# 	stats, *runner_state = runner.run(*runner_state)
# 	end = time.time()

# 	sps = 1/(end-start)*runner.step_batch_size*runner.n_rollout_steps
# 	stats.update({'steps': train_steps, 'sps': sps})