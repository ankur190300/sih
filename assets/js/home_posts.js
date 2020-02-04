{
    // method to submit the form data for new post using AJAX
    //console.log("hello")
    let createPost = function () {
        let newPostForm = $('#new-post-form');

        newPostForm.submit(function (e) {
            e.preventDefault();
            $.ajax({
                type: 'post',
                url: '/posts/create',
                data: newPostForm.serialize(),
                success: function (data) {
                   
                    let newPost = newPostDom(data.data.post);
                    $('#posts-list-container > ul').prepend(newPost);
                    console.log($('.delete-post button', newPost).prop('href'))
                    deletePost($(' .delete-post button', newPost));
                },
                error: function (data) {
                    console.log(err.responseText);
                }

            });
            
            
        });
    }


    // method to create a post in DOM
    let newPostDom = function (post) {
        return $(` <li id="post-${ post._id}">
                      <p>
                           
                           <small>
                             <a class = "delete-post button" href ="/posts/delete/${ post._id}"> X </a>
                           </small>
                           
            
            
                           ${ post.content }

                           <br>

                            <small>

                               ${ post.user.name }

                            </small>
                            <small>
                            
                            <a class="toggle-like-button" data-likes="0"  href="/likes/toggle/?id=<%=post._id%>&type=Post">
                                0 Likes
                            </a>
                     
                            </small>
                            
                       </p>
                              <div class="post-comments">
                
                
                            
                  
                  
                            <form action="/comments/create" method ="POST">
                              <input type="text" name ="content" placeholder="add a comment...">
                              <input type="submit" value="submit">
                              <input type ="hidden" name="post" value="${post._id}">
                            </form>
                            
                    
                    
                    
                              <div class="post-comments-list">
                                <h3>COMMENTS </h3>
                                <ul class="post-comments-${post._id}">
                    
                                  
                         
                                </ul>
                    
                              </div>  
                
                          </div>

           

             </li> `)
    }

    //Method to delete the post from the DOM
    let deletePost = function (deletelink) {

        //console.log($(deletelink).prop('href'));
        $(deletelink).click(function (e) {
            e.preventDefault();
            console.log("hello");
            
            $.ajax({
                type: 'get',
                url: $(deletelink).prop('href'),
                success: function (data) {
                    console.log(data);
                    $(`#post-${data.data.post_id}`).remove();
                },
                error: function (error) {
                    console.log(error.responseText);
                }
            });


        })

    }

    createPost();

    


}


