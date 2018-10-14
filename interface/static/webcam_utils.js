
// Assumes dependencies on webcam.js
let timer = null;
let face_detection_active = false;

// take_snapshot_success_callback = (src, face_found) => {
//   if (face_found) {
//     select_src(src);
//   } else {
//     take_snapshot();
//   }
// };

take_snapshot = () => {
  if (face_detection_active) {
    Webcam.snap( function(data_uri) {
      // Assumes dependencies on animator.js
      twinGANContinuous(data_uri, take_snapshot);
    } );
  }
};

switch_face_detection = () => {
  if (face_detection_active) {
    face_detection_active = false;
    $("#start_face_detection_btn").show();
    $("#pause_face_detection_btn").hide();
  } else {
    face_detection_active = true;
    $("#start_face_detection_btn").hide();
    $("#pause_face_detection_btn").show();
    take_snapshot();
  }
}

// This version takes one image regardless of face detection result.
// take_snapshot = () => {
//   // take snapshot and get image data
//   Webcam.snap( function(data_uri) {
//     // display results in page
//     // document.getElementById('results').innerHTML =
//     //   '<h2>Here is your image:</h2>' +
//     //   '<img src="'+data_uri+'"/>';
//
//     // Assumes dependencies on animator.js
//     select_src(data_uri);
//   } );
// };


detect_face_success_callback = (_unused_src, _unused_face_found) => {
  if (face_detection_active) {
    take_snapshot_for_face_detection();
  }
};

take_snapshot_for_face_detection = () => {
  // take snapshot and get image data
  Webcam.snap( function(data_uri) {
    // Assumes dependencies on animator.js
    detectFace(data_uri, detect_face_success_callback);
  } );
};

start_detect_face = () => {
  face_detection_active = true;
  take_snapshot_for_face_detection();
};
stop_detect_face = () => {
  face_detection_active = false;
  erase_snaps();
};

function attach_webcam () {
   if (!$("#my_camera").size()) {
     setTimeout(attach_webcam, 500); // give everything some time to render
   } else {
      Webcam.set({
        width: 960,
        height: 720,
        image_format: 'jpeg',
        jpeg_quality: 90
      });
     Webcam.attach( '#my_camera' );
   }
 }
 attach_webcam();

function erase_snaps() {
  // $('#visualize_face').hide();
  document.getElementById('visualize_face').src = '';
}
