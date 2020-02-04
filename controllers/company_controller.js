const Job = require('../models/jobs')


module.exports.home =  async function(req, res){

    try{
        
        jobs =  await Job.find({});
        console.log(jobs);
        return res.render('comp_home', {
            title:"Company home", 
            all_jobs:jobs
        })

    }catch(err){
        req.flash('error', err);
        
        return res.redirect('back');

    }
    
}

module.exports.com_jobs = async function(req, res){

    try{
        jobs =  await Job.find({company:req.user.id});
        return res.render('all_jobs', {
            title: "all jobs",
            comp_jobs : jobs
        })
    }catch(err){
        req.flash('error', err);
        
        return res.redirect('back');

    }
    
}