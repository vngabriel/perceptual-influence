import os
import psutil
import subprocess


class SystemMonitor:
    def retrieve_tokens(self, s, line_num):
        tokens_string = s.split("\\n")[line_num].split(" ")
        return filter(lambda x: x != "", tokens_string)

    def execute_bash_command(self, cmd):
        tenv = os.environ.copy()
        tenv["LC_ALL"] = "C"
        bash_command = cmd
        process = subprocess.Popen(
            bash_command.split(), stdout=subprocess.PIPE, env=tenv
        )
        return process.communicate()[0]

    def get_mem_usage(self):
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / 125000.0)

    def get_gpu_info(self):
        gpu_info = []
        try:
            bash_command = "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,count,utilization.gpu,utilization.memory --format=csv"
            output = self.execute_bash_command(bash_command)
            lines = str(output).split("\\n")
            lines.pop(0)

            for l in lines:
                tokens = l.split(", ")
                if len(tokens) > 6:
                    gpu_info.append(
                        {
                            "id": tokens[0],
                            "name": tokens[1],
                            "mem": tokens[3],
                            "cores": tokens[6],
                            "mem_free": tokens[4],
                            "mem_used": tokens[5],
                            "util_gpu": tokens[7],
                            "util_mem": tokens[8],
                        }
                    )
        except OSError:
            raise Exception("GPU device is not available")

        return gpu_info[0]
