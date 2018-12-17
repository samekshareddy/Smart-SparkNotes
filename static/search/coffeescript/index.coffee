$ ->
  $(".input-group").keypress (k) ->
    if k.which == 13 # enter key pressed
      $("#summarize_button").click()
      return false;
  $("#summarize_button").click -> do_summary()


$('#isFull').click -> $(this).toggleClass 'checked'


do_summary = () ->
  query = $("#query_text").val()
  start = $("#start").val()
  isFull = false
  length = $("#len").val() * 1


  if $('#isFull').hasClass 'checked'
    isFull = true

  if true
    $("#search_results_list").empty()
    $("#search_results_list").append("<li> Please wait for the summary </li>")
    $.ajax "get-pdf",
      type: "POST"
      contentType: "application/json; charset=utf-8"
      dataType: "json"
      data: JSON.stringify
        query: query,
        start: start,
        isFull: isFull,
        length: length
      success: (data, stat, xhr) -> print_results data
      failure: (axhr, stat, err) ->
        $("#search_results_list").append("<li>Something bad happened!</li>")



print_results = (result) ->
  console.log(result)
  html = "<p>#{result["content"]}"
  $("#search_results_list").empty()
  $("#search_results_list").append(html)
