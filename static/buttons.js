function get_ideas()
{
  $.ajax({
    type: 'POST',
    url: '/get_ideas',
    data:{}
  }).done(function(response) {
    console.log(response)
    document.getElementById('get_ideas_header').innerHTML = '<br>Try one of these:<br>';
    document.getElementById('get_ideas_result').innerHTML = response;
  });
}

function predict_img()
{
  document.getElementById('get_ideas_header').innerHTML = '';
  document.getElementById('get_ideas_result').innerHTML = '';
  document.getElementById('prediction_result').innerHTML = 'Making prediction...';
  
  var canvas = document.getElementById('Canvas');
  var dataURL = canvas.toDataURL('image/jpg');
  $.ajax({
    type: 'POST',
    url: '/prediction_page',
    data:{
      image: dataURL
      }
  }).done(
    function(response) {
      console.log(response['message'])
      document.getElementById('prediction_result').innerHTML = response['message']
      prediction_response = response
    }
  );
  document.getElementById('correct_or_not_choice').innerHTML = `
    <button
      type="button"
      class="btn btn-default my_button"
      onclick="save_img()"
      style="width:126px">
      Correct!</button>
    <button
      type="button"
      class="btn btn-default my_button"
      onclick="show_form()"
      style="width:126px">
      Incorrect</button>
  `;
}

function show_form()
{
  document.getElementById('fill_actual_label_form').innerHTML = `
  <form method="post">
    <input
      placeholder="what was it?"
      type="string"
      size="21"
      id="drawing_name">
    <button
      class="btn btn-default my_button"
      onclick="save_img(); return false">
      Save</button>
  </form>`
}

function save_img()
{
  var dataURL = document.getElementById('Canvas').toDataURL('image/jpg');
  var drawing_name = prediction_response['predicted_label'];
  var submitted = document.getElementById('drawing_name');
  if(submitted){
    var drawing_name= submitted.value;
  }
  $.ajax({
    type: 'POST',
    url: '/save_img',
    data:{
      'confidence': prediction_response['confidence'],
      'predicted_label': prediction_response['predicted_label'],
      'actual_label': drawing_name,
      'image': dataURL
      }
  }).done(
    function(response) {
      console.log(response)
      document.getElementById('correct_or_not_choice').innerHTML = '<h4>Answer is saved, thank you!</h4>'
      document.getElementById('fill_actual_label_form').innerHTML = ''
    }
  );
}