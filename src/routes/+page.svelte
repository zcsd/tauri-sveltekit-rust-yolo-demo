<script>
    import { onMount } from 'svelte';
    import { invoke } from '@tauri-apps/api/tauri'

    let imageUrl = '/placeholder.svg';
    let resImageUrl = '/placeholder.svg';
    let sigmaValue = 0.1;

    onMount(() => {
        initYoloONNX();
    });

    async function initYoloONNX() {
        try {
            const result = await invoke('init_yolo_onnx');
            if (result == 'OK') {
                console.log("Yolov8 ONNX initialized.");
            } else {
                console.error("Yolov8 ONNX init failed.");
            }
        } catch (error) {
            console.error('Yolo8 ONNX init error:', error);
        }
    }

    async function runYoloDetect() {
        try {
            const result = await invoke('run_yolo_detect');
            if (result) {
                const { encoded, format } = result;
                console.log("Yolov8 detect done.");
                resImageUrl = `data:image/${format};base64,${encoded}`;
            }
        } catch (error) {
            console.error('Yolo8 detect error:', error);
            resImageUrl = '/placeholder.svg';
        }
    }

    async function openImage() {
        try {
            const result = await invoke('open_image');
            if (result) {
                const { encoded, format } = result;
                console.log("Image opened, format:", format);
                imageUrl = `data:image/${format};base64,${encoded}`;
                resImageUrl = '/placeholder.svg'; // reset processed image
            }
        } catch (error) {
            console.error('Image open error or cancel:', error);
        } 
    }

    async function processImage(sigma) {
        sigmaValue = sigma;
        try {
            const result = await invoke('process_image', { sigma: sigma });
            if (result) {
                const { encoded, format } = result;
                console.log("Image processed with sigma:", sigma);
                resImageUrl = `data:image/${format};base64,${encoded}`;
            }
        } catch (error) {
            console.error('Image processed error or cancel:', error);
            resImageUrl = '/placeholder.svg';
        }
    }

    function resetImage() {
        imageUrl = '/placeholder.svg';
        resImageUrl = '/placeholder.svg';
        console.log("Reset image.");
    }
</script>

<div class="min-h-screen bg-gray-100 p-8">
  <div class="max-w-7xl mx-auto">
      <h1 class="text-3xl font-bold text-center mb-8">Tauri (Rust & SvelteKit) Image Demo</h1>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div class="flex flex-col justify-between">
              <div class="bg-white p-4 shadow rounded-lg flex justify-center items-center">
                  <img
                    src={imageUrl}
                    alt="Original image"
                    class="max-w-full h-auto"
                    style="width: 300px; height: 300px; object-fit: cover;"
                    aria-hidden="true"
                  />
              </div>
              <div class="mt-4 flex justify-center">
                  <div class="flex justify-center">
                      <button on:click={openImage} class="bg-green-300 items-center justify-center rounded-md text-sm font-medium hover:bg-green-500 h-10 px-4 py-2 flex-none" style="width: 150px;">
                        Upload
                      </button>
                      <button on:click={resetImage} class="ml-2 bg-green-300 items-center justify-center rounded-md text-sm font-medium hover:bg-green-500 h-10 px-4 py-2 flex-none" style="width: 150px;">
                        Reset
                      </button>
                  </div>
              </div>
          </div>
          <div class="flex flex-col justify-between">
              <div class="bg-white p-4 shadow rounded-lg flex justify-center items-center">
                  <img
                    src={resImageUrl}
                    alt="Processed image"
                    class="max-w-full h-auto"
                    style="width: 300px; height: 300px; object-fit: cover;"
                    aria-hidden="true"
                  />
              </div>
              <!-- 
              <div class="mt-4 flex justify-center">
                <div class="flex items-center space-x-2">
                    <button on:click={runYoloDetect} class="ml-2 bg-green-300 items-center justify-center rounded-md text-sm font-medium hover:bg-green-500 h-10 px-4 py-2 flex-none" style="width: 150px;">
                        YOLO GO!
                    </button>
                </div>
              </div>
              -->
              <div class="mb-2 flex justify-center">
                  <div class="flex items-center space-x-2">
                      
                      <label for="image-slider" class="text-sm font-medium text-gray-700">
                          Sigma
                      </label>
                      <input
                          on:input={(e) => processImage(e.target.value)}
                          id="image-slider"
                          min="0.1"
                          max="10"
                          step="0.1"
                          class="w-64 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" 
                          type="range"
                          value="0.1"
                      />
                      <label for="image-slider" class="text-sm font-medium text-gray-700" style="width: 20px; display: inline-block; text-align: center;">
                        {sigmaValue}
                    </label>
                  </div>
              </div>

          </div>
      </div>
  </div>
</div>
