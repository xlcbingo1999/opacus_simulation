import heapq
def get_schedule_order(waiting_job_selected_sign):
    job_num = len(waiting_job_selected_sign)
    temp_sign_list = [0] * job_num
    heap = []
    for i in range(job_num):
        temp_sign_list[i] = len(waiting_job_selected_sign[i])
        if temp_sign_list[i] > 0:
            for j in range(temp_sign_list[i]):
                heapq.heappush(heap, (-waiting_job_selected_sign[i][j], (i, j)))
    schedule_order = []
    while len(heap) > 0:
        _, (x, y) = heapq.heappop(heap)
        schedule_order.append((x, y))
    return schedule_order

waiting_job_selected_sign = [
	[1, 4, 2, 4],
	[2, 5, 6],
	[2, 4, 6]
]
waiting_job_selected_datablock_identifiers = [
	['a'] * 4,
	['b'] * 3,
	['c'] * 3
]

get_schedule_order(waiting_job_selected_sign, waiting_job_selected_datablock_identifiers)