function isFileImage(file) {
  return file && file['type'].split('/')[0] === 'image';
}

var loader = function(e) {
  let file = e.target.files;
  let output = document.getElementById("selector");
  //console.log(file[0].type);
  //console.log(file[0].type.split('/')[0]);
  if (file[0].type.split('/')[0] === 'image') {
    // let show = "<span> Selected File : </span>" + file[0].name;

    let reader = new FileReader();
    reader.addEventListener("load", function(e) {
      let data = e.target.result;
      let image = document.createElement("img");
      image.src = data;

      output.innerHTML = "";
      output.insertBefore(image, null);
      output.classList.add("image");
    });
    
    reader.readAsDataURL(file[0]);
    // output.innerHTML = show;
    output.classList.add("active");

    // document.getElementsByName("submit").hidden = false;
    document.getElementById("submitBtn").hidden = false;
  }
  else {
    alert('Please select an image.')
  }
}; 

//event listener for input

let fileInput = document.getElementById("file");
fileInput.addEventListener("change",loader); 

// var loader = function(e) {
//   let file = e.target.files;
//   //console.log(file[0].type);
//   //console.log(file[0].type.split('/')[0]);
//   if (file[0].type.split('/')[0] === 'image') {
//     let show = "<span> Selected File : </span>" + file[0].name;

//     let output = document.getElementById("selector");

//     output.innerHTML = show;
//     output.classList.add("active");
//   }
//   else {
//     alert('Please select an image.')
//   }
// }; 