// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use base64::{encode_config, STANDARD};
use std::fs::File;
use std::io::{Read, Cursor};
use rfd::FileDialog;
use serde::Serialize;
use tauri::State;
use std::sync::{Arc, Mutex};
use image::{ImageOutputFormat, io::Reader as ImageReader, imageops::FilterType};
use std::time::Instant;

mod yolo;

#[derive(Serialize)]
struct ImageData {
    encoded: String,
    format: String,
}

struct AppState {
  image_data: Arc<Mutex<Option<Vec<u8>>>>,
  yolo_model: Arc<Mutex<Option<yolo::Model>>>
}

fn main() {
  let app_state = AppState {
    image_data: Arc::new(Mutex::new(None)),
    yolo_model: Arc::new(Mutex::new(None))
  };

  tauri::Builder::default()
    .manage(app_state)
    .invoke_handler(tauri::generate_handler![open_image, process_image, init_yolo_onnx, run_yolo_detect])
    .run(tauri::generate_context!())
    .expect("Error while running Tauri application.");
}
/*   
#[tauri::command]
fn run_yolo_onnx(app_state: State<AppState>) -> Result<ImageData, String> {
  let mut model = match yolo::load_model() {
    Ok(model) => model,
    Err(e) => {
      return Err(format!("Failed to load model: {}", e));
    }
  };

  let image_data = {
    let image_data = app_state.image_data.lock().unwrap();
    image_data.clone().ok_or("No image loaded")?
  };

  let img_vec:opencv::core::Vector<u8> = opencv::core::Vector::from_iter(image_data);
  let mat = opencv::imgcodecs::imdecode(&img_vec, opencv::imgcodecs::IMREAD_UNCHANGED);
  
  let mut mat = mat.unwrap();
  let detections = yolo::detect(&mut model, &mat, 0.5, 0.5);

  if detections.is_err() {
    return Err(format!("Failed to detect: {}", detections.err().unwrap()));
  }

  let detections = detections.unwrap();
  println!("{:?}", detections); 
  yolo::draw_predictions(&mut mat, &detections, &model.model_config);

  let params: opencv::core::Vector<i32> = opencv::core::Vector::default();
    
  //opencv::imgcodecs::imwrite("ttt.jpg", &mat, &params).unwrap();
  // convert the mat to base64 encoded image data
  let mut encoded = opencv::types::VectorOfu8::new();
  opencv::imgcodecs::imencode(".jpg", &mat, &mut encoded, &params).unwrap();
  let base64_string = encode_config(&encoded.to_vec(), STANDARD);

  Ok(ImageData { encoded: base64_string, format: "jpeg".to_string() })
}
*/

#[tauri::command]
fn init_yolo_onnx(app_state: State<AppState>) -> Result<String, String> {
  let model = match yolo::load_model() {
    Ok(model) => model,
    Err(e) => {
      return Err(format!("Failed to load Yolo ONNX model: {}", e));
    }
  };

  // save the model to the app state
  let mut yolo_model = app_state.yolo_model.lock().unwrap();
  *yolo_model = Some(model);

  Ok("OK".to_string())
}

#[tauri::command]
fn run_yolo_detect(app_state: State<AppState>) -> Result<ImageData, String> {
  // get the yolo onnx model from the app state
  let mut model_guard = app_state.yolo_model.lock().unwrap();
  let mut model = model_guard.as_mut().expect("Failed to get Yolo ONNX model from app state.");

  let image_data = {
    let image_data = app_state.image_data.lock().unwrap();
    image_data.clone().ok_or("Failed to get image from app state")?
  };

  let img_vec:opencv::core::Vector<u8> = opencv::core::Vector::from_iter(image_data);
  let mat = opencv::imgcodecs::imdecode(&img_vec, opencv::imgcodecs::IMREAD_UNCHANGED);
  let mut mat = mat.unwrap();

  let start = Instant::now(); 

  let detections = yolo::detect(&mut model, &mat, 0.5, 0.5);
  if detections.is_err() {
    return Err(format!("Failed to detect: {}", detections.err().unwrap()));
  }

  let detections = detections.unwrap();
  let duration = start.elapsed();
  println!("Detection time is: {:?} ms", duration.as_millis());
  println!("{:?}", detections); 
  yolo::draw_predictions(&mut mat, &detections, &model.model_config);

  let params: opencv::core::Vector<i32> = opencv::core::Vector::default();

  //opencv::imgcodecs::imwrite("ttt.jpg", &mat, &params).unwrap();
  // convert the mat to base64 encoded image data
  let mut encoded = opencv::types::VectorOfu8::new();
  opencv::imgcodecs::imencode(".jpg", &mat, &mut encoded, &params).unwrap();
  let base64_string = encode_config(&encoded.to_vec(), STANDARD);

  Ok(ImageData { encoded: base64_string, format: "jpeg".to_string() })
}

#[tauri::command]
fn open_image(app_state: State<AppState>) -> Result<ImageData, String> {
  let file = FileDialog::new()
  .add_filter("image", &["jpg", "jpeg", "png"])
  .set_directory("/")
  .pick_file();

  if let Some(path) = file {
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_lowercase();

    let format = match extension.as_str() {
        "jpg" | "jpeg" => "jpeg",
        "png" => "png",
        _ => return Err("Unsupported format image opened.".to_string())
    };
    
    let mut file = File::open(&path).map_err(|e| e.to_string())?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| e.to_string())?;

    // save the raw image data to the app state
    let mut image_data = app_state.image_data.lock().unwrap();
    *image_data = Some(buffer.clone());

    // return the image data as base64
    let base64_string = encode_config(buffer, STANDARD);
    Ok(ImageData { encoded: base64_string, format: format.to_string() })
  } else {
      Err("User cancel to upload image.".to_string())
  }
}

#[tauri::command]
fn process_image(app_state: State<AppState>, sigma: String) -> Result<ImageData, String> {
  println!("sigma: {}", sigma);
  // convert sigma string to f32
  let sigma = sigma.parse::<f32>().map_err(|e| e.to_string())?;
  let image_data = {
      let image_data = app_state.image_data.lock().unwrap();
      image_data.clone().ok_or("No image loaded")?
  };

  // load image data as image::DynamicImage
  let image = ImageReader::new(Cursor::new(&image_data))
      .with_guessed_format()
      .map_err(|e| e.to_string())?
      .decode()
      .map_err(|e| e.to_string())?;

  // Resize the image if necessary
  let resized_image = if image.width() > 300 || image.height() > 300 {
    image.resize(300, 300, FilterType::Lanczos3)
  } else {
    image
  };

  // Apply blur to the resized image
  let blurred_image = resized_image.blur(sigma);

  // convert the blurred image to Vec<u8>, and write to buffer as jpeg
  let mut buffer = Cursor::new(Vec::new());
  blurred_image.write_to(&mut buffer, ImageOutputFormat::Jpeg(80))
      .map_err(|e| e.to_string())?;

  // return the base64 encoded image data
  let base64_string = encode_config(buffer.into_inner(), STANDARD);
  Ok(ImageData { encoded: base64_string, format: "jpeg".to_string() })
}


