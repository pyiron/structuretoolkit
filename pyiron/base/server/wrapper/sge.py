import pandas
import xmltodict


class SunGridEngineCommands(object):
    @property
    def submit_job_command(self):
        return ['qsub', '-terse']

    @property
    def delete_job_command(self):
        return ['qdel']

    @property
    def enable_reservation_command(self):
        return ['qalter', '-R', 'y']

    @property
    def get_queue_status_command(self):
        return ['qstat', '-xml']

    @staticmethod
    def convert_queue_status(queue_status_output):
        status_dict = xmltodict.parse(queue_status_output)
        df_running_jobs = pandas.DataFrame(status_dict['job_info']['queue_info']['job_list'])
        df_pending_jobs = pandas.DataFrame(status_dict['job_info']['job_info']['job_list'])
        df_merge = df_running_jobs.append(df_pending_jobs, sort=True)
        df_merge.state[df_merge.state == 'r'] = 'running'
        df_merge.state[df_merge.state == 'qw'] = 'pending'
        df_merge.state[df_merge.state == 'Eqw'] = 'error'
        return pandas.DataFrame({'jobid': pandas.to_numeric(df_merge.JB_job_number),
                                 'user': df_merge.JB_owner,
                                 'jobname': df_merge.JB_name,
                                 'status': df_merge.state})
