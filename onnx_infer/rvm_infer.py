import numpy as np
import onnxruntime as rt


# https://github.com/PeterL1n/RobustVideoMatting/blob/53d74c6826735f01f4406b5ca9075eee27bec094/documentation/inference.md?plain=1#L261
class RVMInferEngine:

    def __init__(self, model_path):
        super(RVMInferEngine, self).__init__()
        self.session = rt.InferenceSession(model_path,
                                           providers=['CUDAExecutionProvider'])
        self.io_binding = self.session.io_binding()
        # Request ONNX Runtime to bind and allocate memory on CUDA for 'output'
        self.io_binding.bind_output('pha', 'cuda')
        self.io_binding.bind_output('fgr', 'cuda')
        self.initialized = False
        self.input_shape = None

    def initialize(self, shape):
        # Create tensors on CUDA.
        rec = [
            rt.OrtValue.ortvalue_from_numpy(
                np.zeros([1, 1, 1, 1], dtype=np.float32), 'cuda')
        ] * 4
        downsample_ratio = rt.OrtValue.ortvalue_from_numpy(
            np.asarray([0.6], dtype=np.float32), 'cuda')
        # Set output binding.
        for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
            self.io_binding.bind_output(name, 'cuda')

        self.io_binding.bind_ortvalue_input('r1i', rec[0])
        self.io_binding.bind_ortvalue_input('r2i', rec[1])
        self.io_binding.bind_ortvalue_input('r3i', rec[2])
        self.io_binding.bind_ortvalue_input('r4i', rec[3])
        self.io_binding.bind_ortvalue_input('downsample_ratio',
                                            downsample_ratio)
        self.initialized = True

    def infer(self, image):
        """
        :param image: [batch_size, 3, height, width], data range from 0 to 1, float
        :return: [batch_size, height, width], data range from 0 to 1, float, inferred transparency
        """

        # The following call returns an OrtValue which has data allocated by ONNX Runtime on CUDA
        image = image.astype(np.float32)
        image_ort = rt.OrtValue.ortvalue_from_numpy(image, 'cuda')
        self.io_binding.bind_ortvalue_input('src', image_ort)

        self.session.run_with_iobinding(self.io_binding)
        ort_output = self.io_binding.get_outputs()[0]
        return ort_output.numpy()
