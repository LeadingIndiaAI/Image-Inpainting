function isFileImage(file) {
  return file && file['type'].split('/')[0] === 'image';
}

var check1 = false;
var check2 = false;

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

    check1 = true;
    unhide();
    // document.getElementsByName("submit").hidden = false;
    // document.getElementById("submitBtn").hidden = false;
  }
  else {
    alert('Please select an image.')
    output.classList.remove("image");
    output.classList.remove("active");
    output.innerHTML = "Select Image";
    // output.insertBefore(null, null);
    check1 = false;
    unhide();
  }
}; 

//event listener for input

let fileInput = document.getElementById("file");
fileInput.addEventListener("change",loader); 



var loader2 = function(e) {
  let file2 = e.target.files;
  let output2 = document.getElementById("selector2");
  //console.log(file[0].type);
  //console.log(file[0].type.split('/')[0]);
  if (file2[0].type.split('/')[0] === 'image') {
    //let show2 = "<span> Selected File : </span>" + file2[0].name;

    let reader2 = new FileReader();
    reader2.addEventListener("load", function(e) {
      let data2 = e.target.result;
      let image2 = document.createElement("img");
      image2.src = data2;

      output2.innerHTML = "";
      output2.insertBefore(image2, null);
      output2.classList.add("image");
    });
    
    reader2.readAsDataURL(file2[0]);
    // output.innerHTML = show;
    output2.classList.add("active");

    // document.getElementsByName("submit").hidden = false;
    // document.getElementById("submitBtn").hidden = false;
    check2 = true;
    unhide();
  }
  else {
    alert('Please select an image.')
    output2.classList.remove("image");
    output2.classList.remove("active");
    output2.innerHTML = "Select Mask";
    // output2.insertBefore(null, null);
    check2 = false;
    unhide();
  }
}; 

let fileInput2 = document.getElementById("mask");
fileInput2.addEventListener("change",loader2);

function unhide() {
  if (check1 === true && check2 === true) {
    document.getElementById("submitBtn").hidden = false;
    console.log('TRUE');
    // document.getElementById("submitBtn").style.visibility = 'visible';
  }
  else {
    console.log('FALSE');
    document.getElementById("submitBtn").hidden = true;
    // document.getElementById("submitBtn").style.visibility = 'hidden';
  }
}

$('#submitBtn').click(function() {
  // $("body").css("cursor", "wait");
  // $("button").css("cursor", "wait");
  // $('button').prop('disabled', true);
  // $('label').prop('disabled', true);
  // $('label').prop('disabled', true);
  $('body').css('pointer-events', 'none');
  // $('button').css('pointer-events', 'none');
  $("html").css("cursor", "wait");
  // $('button').prop('disabled', true);
});

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