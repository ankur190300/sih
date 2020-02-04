const Post = require('../models/posts');
const User = require('../models/JavaScript1')


module.exports.home = async function (req, res) {
    // console.log(req.cookies);
    // res.cookie('user_id', 25);

    //returning all the posts to print on the home page
    /*
    Post.find({}, function (err, post) {

        if (err) {
            console.log('not able to show posts from the postsdb');

        }
        else {
            return res.render('home', { title: 'home', posts:post });
        }
    })*/


    //populating the post
    try {
        let posts = await Post.find({})
            .sort('-createdAt')
            .populate('user')
            .populate({
                path: 'comments',
                populate: {
                    path: 'user'
                },
                populate: {
                    path: 'likes'
                }
            })
            .populate('likes');


        //console.log(posts);
        let users =await User.find({},);

        return res.render('home', {
            title: "Codeial | Home",
            posts: posts,
            all_users: users
        });
    }
    catch (err) {
        console.log("error ", err);
    }
}
