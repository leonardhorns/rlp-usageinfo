import asyncio


class Worker:
    def __init__(self, queue: asyncio.Queue, n=10):
        self.n = n
        self.queue = queue
        self.semaphore = asyncio.Semaphore(self.n)

    async def run(self):
        tasks = []
        while True:
            try:
                func = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            tasks.append(asyncio.ensure_future(self.do_work(func)))
        await asyncio.gather(*tasks)

    async def do_work(self, func):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, func)
