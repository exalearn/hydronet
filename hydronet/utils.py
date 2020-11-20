import platform
import os

from tensorflow.python.client import device_lib


def _get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def get_platform_info():
    """Get information about the computer running this process"""

    if hasattr(os, 'sched_getaffinity'):
        accessible = len(os.sched_getaffinity(0))
    else:
        accessible = os.cpu_count()
    return {
        'processor': platform.machine(),
        'python_version': platform.python_version(),
        'python_compiler': platform.python_compiler(),
        'hostname': platform.node(),
        'os': platform.platform(),
        'cpu_name': platform.processor(),
        'n_cores': os.cpu_count(),
        'accessible_cores': accessible,
        'tf_xla_flags': os.environ.get('TF_XLA_FLAGS', ''),
        'gpus': _get_available_gpus()
    }

