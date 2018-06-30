function predict_img()
{
    document.getElementById('get_ideas_header').innerHTML = '';
    document.getElementById('get_ideas_result').innerHTML = '';
    document.getElementById("prediction_result").innerHTML = "Making prediction...";

    var canvas = document.getElementById("Canvas");
    var dataURL = canvas.toDataURL('image/jpg');
    $.ajax({
      type: "POST",
      url: "/prediction_page",
      data:{
        image: dataURL
        }
    }).done(function(response) {
      console.log(response)
      document.getElementById("prediction_result").innerHTML = response
    });
}

function get_ideas()
{
    $.ajax({
      type: "POST",
      url: "/get_ideas",
      data:{}
    }).done(function(response) {
      console.log(response)
      document.getElementById("get_ideas_header").innerHTML = '<br>Try one of these:<br>';
      document.getElementById("get_ideas_result").innerHTML = response;
    });
    
}